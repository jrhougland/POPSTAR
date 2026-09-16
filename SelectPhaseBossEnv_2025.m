classdef SelectPhaseBossEnv_2025 < rl.env.MATLABEnvironment
    properties
        NumberOfSteps;
        InitialPhase = deg2rad(-90);
        RewardCap = 10;
        Phase_tolerance = deg2rad(5);
        subj_id;
        session;
        
    end
    
    properties(Access=protected)
        bd;
        stim;
        sc;
        record;
        IsDone = false;
        Trial = 0;
        phases = [];
        EMG_pause = 0.4;
        Episode;
        save_flag;
        temp_MEP = [];
        Actions = [];
        reward = [];
 
    end
    
    methods
        function this = SelectPhaseBossEnv_2025(bd, stim, sc, record, save_flag, session)
            ObservationInfo = rlNumericSpec([2 1]);
            ObservationInfo.Name = 'Observation';
            ObservationInfo.Description = 'phase, mep_amplitude';
            
            ActionInfo = rlFiniteSetSpec(1:8);
            ActionInfo.Name = 'Action';
            
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            this.bd = bd;
            this.stim = stim;
            this.sc = sc;
            this.record = record;
            this.Episode = 0;
            this.save_flag = save_flag;
            this.session = session;
        end

        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            this.Trial = this.Trial + 1;
            LoggedSignals = [];
            
            phase = (Action - 5) * 0.25 * pi;
            this.phases = [this.phases, phase];
            
            
            if strcmp(this.bd.armed, 'no')
                this.bd.triggers_remaining = 1;
                this.bd.configure_time_port_marker([0 1 0]);
                this.bd.alpha.phase_target(1) = phase;
                this.bd.alpha.phase_plusminus(1) = this.Phase_tolerance;
                
                ITI = this.sc.iti(1) + (this.sc.iti(2) - this.sc.iti(1)) * rand(1,1);
                pause(ITI);
                
                this.bd.arm;
            end
            
            while this.bd.triggers_remaining == 1
                pause(0.01);
            end
            
            mep_bug = false;
            while ~mep_bug
                if this.bd.triggers_remaining == 0
                    trial_index = length(this.record.emg) + 1;
                    pause(this.EMG_pause);
                    try
                        this.record.emg(trial_index).signal = this.bd.mep(this.sc.mep_channel, 100, 100) * 5;
                    catch
                        disp('MEP missed. Retrying...');
                        continue;
                    end
                    
                    this.record.emg(trial_index).signal = this.record.emg(trial_index).signal - mean(this.record.emg(trial_index).signal( ...
                        this.sc.baselineTimeWindow(1) <= this.sc.emgTimeAxis & ...
                        this.sc.emgTimeAxis <= this.sc.baselineTimeWindow(2)));
                    
                    temp = extractMEP(this.record.emg(trial_index).signal, this.sc.emgTimeAxis, this.sc.mepTimeWindow, this.sc.baselineTimeWindow);
                    current_MEP = temp.amplitude;
                    mep_bug = true;
                    this.bd.disarm;
                end
            end
            
            this.temp_MEP = [this.temp_MEP, current_MEP];
            this.Actions = [this.Actions, Action];
           
            % Implement weighted average giving more importance to recent MEPs
            num_MEPs = length(this.temp_MEP);
                if num_MEPs > 1
                    % Create weights that increase linearly for more recent MEPs
                    weights = linspace(0.5, 1.5, num_MEPs);
    
                    % Normalize weights to sum to 1
                    weights = weights / sum(weights);
    
                    % Calculate weighted average
                    avg_MEP = sum(weights .* this.temp_MEP);
                else
                    avg_MEP = this.temp_MEP;
                end
          
            
            Reward = (current_MEP - avg_MEP*1.2);
            this.reward = [this.reward, Reward];
            
            Observation = [phase; current_MEP];
            
            fprintf('Action: %d (Phase: %.2f rad), MEP: %.2f, Reward: %.2f\n', ...
                Action, phase, current_MEP, Reward);
            
            IsDone = this.Trial >= this.NumberOfSteps;
            this.IsDone = IsDone;
        end

        function InitialObservation = reset(this)
            this.Trial = 0;
            this.IsDone = false;
            this.Episode = this.Episode + 1;
            InitialObservation = [this.InitialPhase; 0];
        end
        
        function saveData = saveMEP(this)
            this.record.MEP = this.temp_MEP;
            this.record.phases = this.phases;
            this.record.Actions = this.Actions;
            this.record.reward = this.reward;
            
            if this.save_flag
                path.save = ['Z:\Experimental Data\2025-04 POPSTAR\POPSTAR_' num2str(this.subj_id, '%03.0f') '\' this.session.protocol];
                if ~exist(path.save, 'dir')
                    mkdir(path.save);
                end
                filename = fullfile(path.save, ['POPSTAR_' num2str(this.subj_id, '%03.0f') '_training.mat']);
                saveData = this.record;
                save(filename, 'saveData');
                disp('Record saved.');
            end
        end
        
        function setNumStepsPerEp(this, numSteps)
            this.NumberOfSteps = numSteps;
        end
    end
end


cd(root);
addpath(genpath('toolboxes'));

eeglab('nogui');

% load data
subject_data = module_read_neurone(...
        fullfile(data_load, sub_id), ...
        sessionPhaseNumber = sess_id);

channel_labels = fieldnames(subject_data.signal)';
channel_labels(end-1:end) = [];
num_channels = length(channel_labels);
N = length(subject_data.signal.(channel_labels{1}).data);
rs_EEG = zeros(N, num_channels);
for i = 1:length(channel_labels)
    rs_EEG(:,i) = subject_data.signal.(channel_labels{i}).data;
end

EEG_data = eeg_emptyset();
EEG_data.data = rs_EEG';
EEG_data.srate = subject_data.properties.samplingRate;
EEG_data.nbchan = size(rs_EEG, 2);
EEG_data.pnts = size(rs_EEG, 1);
EEG_data.trials = 1;
EEG_data.xmin = 0;
EEG_data.xmax = (EEG_data.pnts - 1) / EEG_data.srate;
EEG_data.times = linspace(EEG_data.xmin, EEG_data.xmax, EEG_data.pnts) * 1000; % in ms

% Create chanlocs structure compatible with EEGlab
for i = 1:length(channel_labels)
    EEG_data.chanlocs(i).labels = channel_labels{i};
end

% Save data
EEG = eeg_checkset(EEG_data);  % optional sanity check
pop_saveset(EEG, 'filename', [sub_id, '.set'], 'filepath', [data_save, '\', sub_id]);
function deleteCaimanTempFiles(varargin)
%deleteCaimanTempFiles Delete temporary files from Caiman analysis
%   Remove temporary Caiman files from a directory
%  input1 ... directory for processing
% if no input is provided, the directory is selected by UI dialog
% the script determines a list of files to be deleted, according to the following criteria:
% 1. all files ending with .mmap, .npz or .npy
% 2. .tif files without 'remFrames' in the name
% 3. .png files with '_Components_' in the name


if nargin == 1
    start_dir =  varargin{1};
else
    start_dir = uigetdir(pwd, 'Please select the start folder');
end

% get folder content
folder_content = dir(start_dir);

% remove directories
folder_content([folder_content.isdir]) = [];

% select files for deletion
files_to_delete = {};
counter = 0;
size_counter = 0;
for ix = 1:numel(folder_content)
    % all files ending with .mmap, .npz or .npy
    if endsWith(folder_content(ix).name, '.mmap') | endsWith(folder_content(ix).name, '.npz') | endsWith(folder_content(ix).name, '.npy')
        counter = counter + 1; files_to_delete{counter, 1} = folder_content(ix).name;
        size_counter = size_counter + folder_content(ix).bytes;
    % .tif files without 'remFrames' in the name
    elseif endsWith(folder_content(ix).name, '.tif') & ~strfind(folder_content(ix).name, 'remFrames')
        counter = counter + 1; files_to_delete{counter, 1} = folder_content(ix).name;
        size_counter = size_counter + folder_content(ix).bytes;
    % .png files with '_Components_' in the name
    elseif endsWith(folder_content(ix).name, '.png') & strfind(folder_content(ix).name, '_Components_')
        counter = counter + 1; files_to_delete{counter, 1} = folder_content(ix).name;
        size_counter = size_counter + folder_content(ix).bytes;
    end
end

if numel(files_to_delete) < 1
    disp('No files matching criteria. Exiting.')
    return
end

fprintf('\n\nList of files to be deleted. Please check carefully!\n')

fprintf('Folder:\n%s\n\n', start_dir)
fprintf('Files:\n')
for ix = 1:numel(files_to_delete)
    fprintf('%s\n', files_to_delete{ix})
end

x = input('Type yes to poceed with deletion: ', 's');

if strcmp(x, 'yes')
    for ix = 1:numel(files_to_delete)
        delete(fullfile(start_dir, files_to_delete{ix}))
    end
    fprintf('\nDone. You saved %1.1f GB disk space!\n\n', size_counter/1000000000)
else
    disp('Skipping deletion of temporary files')
end

end


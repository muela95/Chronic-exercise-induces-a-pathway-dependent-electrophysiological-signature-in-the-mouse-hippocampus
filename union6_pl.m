% Define file paths
%sedFile = 'D:/Laboratorio/Registros/Experimental-ejercicio/sed/Schff8-sed.mat';
%runFile = 'D:/Laboratorio/Registros/Experimental-ejercicio/run/Schff8-run.mat';
%outputFile = 'D:/Laboratorio/Registros/Experimental-ejercicio/Schff8-todo-nuevo.mat';

sedFile = 'D:/Laboratorio/Registros/Experimental-ejercicio/para_perforante_lateral/pl6-sed.mat';
runFile = 'D:/Laboratorio/Registros/Experimental-ejercicio/para_perforante_lateral/pl6-run.mat';
outputFile = 'D:/Laboratorio/Registros/Experimental-ejercicio/PL6-todo.mat';

% Load the sedentary data
sedData = load(sedFile);
fprintf('Loaded sedentary data with %d cells\n', numel(sedData.G.s));

% Load the running data
runData = load(runFile);
fprintf('Loaded running data with %d cells\n', numel(runData.G.s));

% Create a new structure for the combined data
G = struct();

% Initialize the cell array for G.s as 2x8
G.s = cell(2, 6);

% Copy sedentary data vectors to first row
for i = 1:6
    % If sedData.G.s is a cell array, extract the vector
    G.s{1, i} = sedData.G.s{i};
end

% Copy running data vectors to second row
for i = 1:6
    G.s{2, i} = runData.G.s{i};
end

% Display information about the combined data
fprintf('Combined data structure created:\n');
fprintf('- Size of G.s: %dx%d\n', size(G.s,1), size(G.s,2));

% Save the combined data to the output file
save(outputFile, 'G');
fprintf('Combined data saved to %s\n', outputFile);

% Verify the saved data
verifyData = load(outputFile);
fprintf('Verification: Loaded combined data with size %dx%d\n', size(verifyData.G.s,1), size(verifyData.G.s,2));

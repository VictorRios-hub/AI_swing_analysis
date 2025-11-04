% Load the .mat file
load('golfDB_remaster.mat');

% Vérifier si le champ 'field' existe
if isfield(votre_structure, 'field')
    % Récupérer le contenu du champ 'field'
    fieldData = votre_structure.field;

    % Identifier les indices des lignes contenant la chaîne 'down-the-line'
    indices = contains(fieldData, 'down-the-line');

    % Filtrer les lignes
    fieldDataFiltered = fieldData(indices);

    % Mettre à jour la structure avec les lignes filtrées
    votre_structure.field = fieldDataFiltered;

    % Sauvegarder la structure mise à jour dans le même fichier .mat
    save('votre_fichier.mat', 'votre_structure');
else
    disp('Le champ ''field'' n''existe pas dans votre structure.');
end
%skin segmentation
data=load('data.txt');
X=data(:,1:3);
Y=data(:,4);

% Au lieu de 1 et 2 comme classes, nous définissons les classes comme 0 et 1 0-> skin
% 1-> non
  Y(Y==1) = 0;
  
  Y(Y==2) = 1;
plotData(X,Y)
legend('skin',' ', 'Not skin')

[m, n] = size(X);

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);

lambda = 0.5;

% Calculer et afficher le coût initial et le gradient
[cost, grad] = costFunction_team_10(initial_theta, X, Y, lambda);
     fprintf('Coût en thêta initial (zéros): %f\n', cost);
  fprintf('Gradient en thta initial  (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgramme en pause. Appuyez sur Entrée pour continuer.\n');
pause;

% Gradient utilisant fminunc
%definition des paramétre d'optimisation 
% gradient objet = on
%max itération = 500 
options = optimset('GradObj', 'on', 'MaxIter', 500);

%minimise avec les paramètres d'optimisation spécifiés dans la structure options
[theta, cost] = fminunc(@(T)(costFunction_team_10(T, X, Y, lambda)), initial_theta, options);
%affichage des valeur minimale de csot et theta dans notre travail 
fprintf('Coût chez theta trouvé par fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Calculer la précision sur notre ensemble de formation
p = predict_team_10(theta, X);
fprintf('Précision de l ensemble d entraînement: %f %\n', mean(double(p == Y)) * 100);

fprintf('\nProgramme en pause. Appuyez sur Entrée pour continuer.\n');
%test
pause;
load('test.txt')
prob = sigmoid_team_10(test * theta);
fprintf('predection \n');
disp('Format: B, G, R, probability_that_it_is_skin, probability_that_it_is_not_skin');
disp([test(:,[2,3,4]) (1-prob) prob ]);




%COSTFUNCTION calcule le cost (l'erreur ) et le gradient de la regression
%logiqtique 
% J = COSTFUNCTION(Theta, X, y) calcule le cost 
% initialisation de qlq variable a utiliser 


                  %************ la fonction ********%
function [J, gradient] = costFunction_team_10(Theta, X, Y, lambda)

gradient = zeros(size(Theta)); % intialiser le gradient a 0 

J = 0; %initialiser le taux d'erreur a 0

m = length(Y); % nombre d'exemples de formation (dans notre cas 245057)
h=sigmoid_team_10(X*Theta); %appele a la fonction sigmoid pour déduire l'hypotése 

%cost(h(x),y)

p1= log(h')*(-Y); %y=1 
p2= log(1.-h')*(1.-Y); %y=0

J=1/m*(p1-p2) + lambda/(2*m)*(sum(Theta.^2)-(Theta(1)^2)); %affecter a J sa valeur 

size_Theta = size(Theta);
rows_Theta = size_Theta(1);

%Mise en œuvre vectorisée
tmp = (h-Y)'*X;
gradient = 1/m*(tmp);

%application de la régularisation vectorisé 
tmp_Theta = Theta;
tmp_Theta(1) = 0;
reg = lambda/m .* tmp_Theta;

gradient = gradient + reg';%ajouter la formule de regularisation pour ne pas tombez dans un over fitting 
for t = 1:rows_Theta
  gradient(t) = 1/m*sum((h-Y)'*X(:,t)); %%remplissage de gradient
  
end  
end

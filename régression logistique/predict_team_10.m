% PREDICT Prédisez si l'étiquette est 0 ou 1 en utilisant la logistique apprise
% paramètres de régression thêta
% p = PREDICT (thêta, X) calcule les prédictions pour X en utilisant un
%Seuil en à 0,5 (c'est-à-dire, si sigmoïde (thêta '* x)> = 0,5, prédire 1)
%si nn  y=0
                   %******************* la fonction *********************%
                   
function pred = predict_team_10(theta, X)
  s = size(X, 1); % Number of training examples
  pred = zeros(s, 1);
prediction = sigmoid_team_10(X*theta); %calcule de l'hypotése 
for j=1:s
  if (prediction(j) >= 0.5)
    pred(j)=1;
  
  end
end
end

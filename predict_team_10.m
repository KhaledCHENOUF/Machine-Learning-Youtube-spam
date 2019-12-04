% PREDICT Pr�disez si l'�tiquette est 0 ou 1 en utilisant la logistique apprise
% param�tres de r�gression th�ta
% p = PREDICT (th�ta, X) calcule les pr�dictions pour X en utilisant un
%Seuil en � 0,5 (c'est-�-dire, si sigmo�de (th�ta '* x)> = 0,5, pr�dire 1)
%si nn  y=0
                   %******************* la fonction *********************%
                   
function pred = predict_team_10(theta, X)
  s = size(X, 1); % Number of training examples
  pred = zeros(s, 1);
prediction = sigmoid_team_10(X*theta); %calcule de l'hypot�se 
for j=1:s
  if (prediction(j) >= 0.5)
    pred(j)=1;
  
  end
end
end

% SIGMOID Calculer la fonction sigmo�de
% J = SIGMOID (z) calcule la fonction sigmo�de / logistique de z

                %*************la fonction ************%

function g = sigmoid_team_10(z)
g = zeros(size(z));%initialiser le g
    g = 1./(1+exp(-1.*z)); %appliquer la formule de la fonction sigmide
end
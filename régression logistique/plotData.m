
% PLOTDATA Trace les points de donn�es X et y dans une nouvelle figure
% PLOTDATA (x, y) trace les points de donn�es avec + pour les exemples positifs
% et o pour les exemples n�gatifs. 
                  %********** la fonction *************%

function plotData(X,Y)
% Creation d'une nouvelle figure 
figure; hold on;
% trouver les indices positif et n�gatif de l'exemple
p = find(Y==1);
n = find(Y==0);
% Ploter l'exemple

plot(X(p, [1]),X(p, [2,3]) ,'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(n, [1]),X(n, [2,3]) ,'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold off;

end

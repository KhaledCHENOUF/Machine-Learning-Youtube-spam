%% ================ Part 1: Chargement des donn�es ================
Inputs = Input;
Targets = Output;

%% ================ Part 2: Cr�ation de r�seau ================
SizeLayer = 10;
net = fitnet(SizeLayer);

%% ================ Part 3: Division des donn�es ================
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%% ================ Part 4: Entrainer le r�seau ================
[net,tr] = train(net,Inputs,Targets);

%% ================ Part 5: Tester le r�seau ================
outputs = net(Inputs);
Errors = gsubtract(outputs,Targets);
Performances = perform(net,Targets,outputs)

%% ================ Part 6:  Afficher le r�seau ================
view(net)


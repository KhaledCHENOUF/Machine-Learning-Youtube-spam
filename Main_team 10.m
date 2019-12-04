%% ================ Part 1: Chargement des données ================
Inputs = Input;
Targets = Output;

%% ================ Part 2: Création de réseau ================
SizeLayer = 10;
net = fitnet(SizeLayer);

%% ================ Part 3: Division des données ================
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%% ================ Part 4: Entrainer le réseau ================
[net,tr] = train(net,Inputs,Targets);

%% ================ Part 5: Tester le réseau ================
outputs = net(Inputs);
Errors = gsubtract(outputs,Targets);
Performances = perform(net,Targets,outputs)

%% ================ Part 6:  Afficher le réseau ================
view(net)


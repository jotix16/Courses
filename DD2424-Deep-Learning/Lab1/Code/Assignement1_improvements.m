    % Assignement Bonus Points
    %% Load all Data
    [X_train, Y_train, X_valid, Y_valid,X_test, Y_test] = loadFullBatch();
    d = size(X_train,1);
    K = size(Y_train,1);
    
    %% Small Data
%     [X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
%     [X_valid, Y_valid, y_valid] = LoadBatch('data_batch_2.mat');
%     [X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
%     d = size(X_train,1);
%     K = size(Y_train,1);
%     
    %% Preprocess
    X_train = Preprocess(X_train);
    X_test = Preprocess(X_test);
    X_valid = Preprocess(X_valid);
    
    %% Training
    % Initialize Params
    std = 0.01;
    W = randn(K,d)*std;
    b = randn(K,1)*std;
    Wsvm = randn(K,d)*std;
    bsvm = randn(K,1)*std;
    
    % Hyperparameters
    lambda = 0.1;
    GDparams.n_epoch = 100;
    GDparams.n_batch= 100;
    GDparams.eta = 0.001;
    
    % Debugging information
    svm_acc = zeros(1,GDparams.n_epoch);
    crosse_acc = zeros(1,GDparams.n_epoch);
    
    % Training Loop (Simultaneus SVM and Cross entropy)
    for i = 1:GDparams.n_epoch
        % Learn Rate Decay
        if mod(i,5) == 0
           GDparams.eta = GDparams.eta*0.9;
        end
        fprintf("Epoche: " +num2str(i)+"\n")
        % Shuffling
        mix = randperm(size(X_train,2));
        X_train = X_train(:,mix);
        Y_train = Y_train(:,mix);
        % Cross entropy
        [W, b] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda); 
        % SVM
        [Wsvm, bsvm] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda,true);
        
        % Debugging info
        crosse_acc(i) = ComputeAccuracy(X_valid, Y_valid, W, b);
        svm_acc(i) = ComputeAccuracy(X_valid, Y_valid, Wsvm, bsvm,true);
    end
    
    %% CrossEntropy vs SVM
    plot(1:GDparams.n_epoch, crosse_acc, 1:GDparams.n_epoch, svm_acc)
    title("Accuracy on validation set for"+": lambda="+num2str(lambda)+", nr epochs="+num2str(GDparams.n_epoch)+", nr batch="+num2str(GDparams.n_batch)+", eta="+num2str(GDparams.eta))
    xlabel('epoch nr')
    ylabel('Accuracy')
    legend("Cross Entropy loss function","SVM loss function")
    
    acc_train = ComputeAccuracy(X_train, Y_train, W, b);
    acc_valid = ComputeAccuracy(X_valid, Y_valid, W, b);
    acc_test = ComputeAccuracy(X_test, Y_test, W, b);
    
    acc_train_svm = ComputeAccuracy(X_train, Y_train, Wsvm, bsvm,true);
    acc_valid_svm = ComputeAccuracy(X_valid, Y_valid, Wsvm, bsvm,true);
    acc_test_svm = ComputeAccuracy(X_test, Y_test, Wsvm, bsvm,true);

    fprintf("The cost function for"+" (lambda="+num2str(lambda)+", nr epochs="+num2str(GDparams.n_epoch)+", nr batch="+num2str(GDparams.n_batch)+", eta="+num2str(GDparams.eta)+") is:\n")
    disp(["acc_train" "acc_valid" "acc_test";acc_train acc_valid acc_test])
    disp(["acc_train" "acc_valid" "acc_test";acc_train_svm acc_valid_svm acc_test_svm])


    %% Ensamble
    % Accuracy for the ensamble of 5-NNs
    acc = ensamble_acc(X_test, Y_test)
    
    
    
    
function acc = ensamble_acc(X, y)
    example1 = matfile('saveA1.mat');
    P1 = EvaluateClassifier(X, example1.W, example1.b);
    [~,classes1] = max(P1);
    C1 = full(ind2vec(classes1));

    example2 = matfile('saveA2.mat');
    P2 = EvaluateClassifier(X, example2.W, example2.b);
    [~,classes2] = max(P2);
    C2 = full(ind2vec(classes2));

    example3 = matfile('saveA3.mat');
    P3 = EvaluateClassifier(X, example3.W, example3.b);
    [~,classes3] = max(P3);
    C3 = full(ind2vec(classes3));

    example4 = matfile('saveA4.mat');
    P4 = EvaluateClassifier(X, example4.W, example4.b);
    [~,classes4] = max(P4);
    C4 = full(ind2vec(classes4));

    example5 = matfile('saveA5.mat');
    P5 = EvaluateClassifier(X, example5.W, example5.b);
    [~,classes5] = max(P5);
    C5 = full(ind2vec(classes5));

    C = C1+C2+C3+C4+C5;
    [~,classes] = max(C);

    if size(y,1)>1
        y=vec2ind(y);
    end
    acc = sum(classes==y)/length(y);
end
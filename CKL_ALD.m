function [expansionCoefficient,biasCoefficient,dictionaryIndex,learningCurve,identOutlier] = ...
    CKL_ALD(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,regularizationFactor2,forgettingfactor,th1,flagLearningCurve,ks,epson_updt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Desenvolvido por Michael Santos Duarte
%PPGETI
%Ago 15, 2018
%
%Descrição:
%Verificar os resultados do algortimo CSKL - Correntropy Sparse Kernel Learning
%Recursivo
%ALD
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%=========CKL===================

% INITIALIZATION

% memory initialization
[inputDimension,trainSize] = size(trainInput);
testSize = length(testTarget);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget(1,:).^2);
else
    learningCurve = [];
end

% Correntropy Funcion
wt = @(e,ks) exp(-(e).^2./(2*ks^2)) ./ (sqrt(2*pi)*ks^3); % Weighted terms
%wt = @(e,ks) tanh(e);

% Init
pe(1) = 1; % Initial value weighted terms
I = ones(1,1); % vector ones

% UPDATE OF THE CKL 

% % LEVEL 1
K = ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel); % Kernel matrix
Q = diag(1./((forgettingfactor^1)*regularizationFactor*pe)); % diagonal matrix
K_matrix = 1/K;
H = K + Q; % matrix symmetric positive-definite invertible
P = pinv(H); % inverse matrix
A = 1;
expansionCoefficient = P*(trainTarget(1) - ((I*I'*P*trainTarget(1))./(I'*trainTarget(1)*I)));% lagrange
biasCoefficient = (I'*P*trainTarget(1))./(I'*trainTarget(1)*I); % bias

% end level 1  

% LEVEL 2
predictionError(1) = trainTarget(1) - expansionCoefficient'*ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel) - biasCoefficient(1); % error of predicted values
pe(1) = wt(predictionError(1),ks); % new weighted terms
aux_pe(1) = pe(1);

Q = diag(1./((forgettingfactor^1)*regularizationFactor*pe)); % diagonal matrix
K_matrix = 1/K;
H = K + Q; % matrix symmetric positive-definite invertible
P = pinv(H); % inverse matrix
P_matrix = 1/H;
expansionCoefficient = P*(trainTarget(1) - ((I*I'*P*trainTarget(1))./(I'*trainTarget(1)*I)));% lagrange
biasCoefficient = (I'*P*trainTarget(1))./(I'*trainTarget(1)*I); % bias

% end level 2

% DICTIONARY
dictSize = 1;
dictionaryIndex(dictSize) = 1;
f_vector(dictSize) = 1;

predictionVar =  regularizationFactor2 + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel);

ALD = predictionVar;

th1Sq = th1^2; % is the sparsity level parameter

% end dictionary
    
for i = 2:trainSize
        
    %Calc the Conditional Information    
    k_vector = ker_eval(trainInput(:,i),trainInput(:,dictionaryIndex),typeKernel,paramKernel);    
    
    % APPROXIMATE LINEAR DEPENDENCE TEST
    f_vector = K_matrix*k_vector;
    
    ktt = ker_eval(trainInput(:,i),trainInput(:,i),typeKernel,paramKernel);
        
    predictionVar = regularizationFactor2 + ktt - k_vector'*f_vector;
	
    ALD(i) = predictionVar;
    
    %UPDATE MATRIX GRAM
    if (ALD(i) < th1Sq)
        if flagLearningCurve==1, learningCurve(i) = learningCurve(i-1); end
        predictionError(i) = trainTarget(i) - expansionCoefficient'*ker_eval(trainInput(:,i),trainInput(:,dictionaryIndex),typeKernel,paramKernel) - biasCoefficient(1); % error of predicted values       
        aux_pe(i) = 1;
        pe(i) = wt(predictionError(i),ks); % new weighted terms
        continue;
    end
    
    % Updating

    s = 1/predictionVar;
    K_tmp = zeros(dictSize+1,dictSize+1);
    K_tmp(1:dictSize,1:dictSize) = K_matrix + f_vector*f_vector'*s;
    K_tmp(1:dictSize,dictSize+1) = -f_vector*s;
    K_tmp(dictSize+1,1:dictSize) = K_tmp(1:dictSize,dictSize+1)';
    K_tmp(dictSize+1,dictSize+1) = s;
    K_matrix = K_tmp;

    Kt = zeros(dictSize+1,dictSize+1);
    Kt(1:dictSize,1:dictSize) = K;
    Kt(1:dictSize,dictSize+1) = k_vector;
    Kt(dictSize+1,1:dictSize) = k_vector';
    Kt(dictSize+1,dictSize+1) = ktt;
    K = Kt;

    P_matrix = P;
    z_vector = P_matrix*k_vector;

    pe(i) = 1; % value weighted terms

    theta = 1/((forgettingfactor^i)*regularizationFactor*pe(i));

    predictionVar2 = theta +...
    ker_eval(trainInput(:,i),trainInput(:,i),typeKernel,paramKernel) -...
    z_vector'*k_vector;

    %update Q_matrix
    s = 1/predictionVar2;
    P_tmp = zeros(dictSize+1,dictSize+1);
    P_tmp(1:dictSize,1:dictSize) = P_matrix + z_vector*z_vector'*s;
    P_tmp(1:dictSize,dictSize+1) = -z_vector*s;
    P_tmp(dictSize+1,1:dictSize) = P_tmp(1:dictSize,dictSize+1)';
    P_tmp(dictSize+1,dictSize+1) = s;
    P_matrix = P_tmp;
    P = P_matrix;

    dictSize = dictSize + 1;
    disp(['ALD-Dictsize...',num2str(dictSize)])
    dictionaryIndex(dictSize) = i;     
    f_vector(dictSize) = 1;

    % LEVEL 1

    I = ones(dictSize,1); % vector ones

    auxTarget = trainTarget(dictionaryIndex);   

    % updating coefficients       
    expansionCoefficient = (P*(auxTarget - ((I*I'*P*auxTarget)./(I'*auxTarget*I))));% lagrange
    biasCoefficient = (I'*P*auxTarget)./(I'*auxTarget*I); % bias       

    % LEVEL 2

    for j = 1:dictSize

        predictionError(dictionaryIndex(j)) = trainTarget(dictionaryIndex(j)) - expansionCoefficient'*ker_eval(trainInput(:,dictionaryIndex(j)),trainInput(:,dictionaryIndex),typeKernel,paramKernel) - biasCoefficient(1); % error of predicted values
        aux_pe(dictionaryIndex(j)) =  pe(dictionaryIndex(j));
        pe(dictionaryIndex(j)) = wt(predictionError(dictionaryIndex(j)),ks); % new weighted terms

        if (abs((1/pe(dictionaryIndex(j)))-(1/aux_pe(dictionaryIndex(j)))) >= epson_updt)

            v = zeros(dictSize,1);
            s = zeros(dictSize,1);

            v(j) = (1/((forgettingfactor^dictionaryIndex(j))*regularizationFactor)); % Vector Nx1
            s(j) = (1/pe(dictionaryIndex(j)))-(1/aux_pe(dictionaryIndex(j))); % Vector Nx1

            P = P - (P*v*s'*P)/(1+s'*P*v);
        end
    end

    expansionCoefficient = P*(auxTarget - ((I*I'*P*auxTarget)./(I'*auxTarget*I)));% lagrange
    biasCoefficient = (I'*P*auxTarget)./(I'*auxTarget*I); % bias

    % TESTING
    if flagLearningCurve
        y_te = zeros(testSize,1);
        for j = 1:testSize
            y_te(j) = expansionCoefficient'*ker_eval(testInput(:,j),trainInput(:,dictionaryIndex),typeKernel,paramKernel) + biasCoefficient(1);
        end
        err = testTarget - y_te;
        learningCurve(i) = 10*log(mean(err.^2));

        disp(['Learning/Dictionary/Error # ',num2str(i.'),'/',num2str(dictSize.'),'/',num2str(learningCurve(i))]);
    end
end

identOutlier = pe;

%======end CKL===========

if flagLearningCurve
    figure;
    plot(learningCurve);
    set(gca, 'FontSize', 14);
    set(gca, 'FontName', 'Arial');
    xlabel('Step'),ylabel('Mean Squared Error (dB)');
    title('MSE (dB) one step ahead');
end

return
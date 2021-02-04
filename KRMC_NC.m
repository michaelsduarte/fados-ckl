function [expansionCoefficient,dictionaryIndex,learningCurve] = ...
    KRMC_NC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,forgettingfactor,toleranceDistance,tolerancePredictError,flagLearningCurve,ks)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function KRMC-NC
%kernel recursive maximum correntropy using novelty criterion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and 
%               trainSize is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%testInput:     testing input, inputDimension*testSize, testSize is the number of the test data
%testTarget:    desired signal for testing testSize*1
%
%typeKernel:    'Gauss', 'Poly'
%paramKernel:   h (kernel size) for Gauss and p (order) for poly
%
%regularizationFactor: regularization parameter in Newton's recursion
%
%
%th1: threshold in NC
%
%flagLearningCurve:    control if calculating the learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%baseDictionary:            dictionary stores all the bases centers
%expansionCoefficient:      coefficients of the kernel expansion
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% memeory initialization
[inputDimension,trainSize] = size(trainInput);
testSize = length(testTarget);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

% wt = @(e,ks) exp(-(e).^2./(2*ks^2)); % Weighted terms
% skew = -.5;
wt = @(e,ks,skew) 2*(exp(-(e).^2./(2*ks^2)))*(.5*(1+erf(skew*e./(ks*sqrt(2))))); % Weighted terms

k_vector = ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel);
Q_matrix = 1/(regularizationFactor*forgettingfactor*ks^2 + k_vector);
expansionCoefficient = Q_matrix*trainTarget(1);

% dictionary
dictionaryIndex = 1;
dictSize = 1;

predictionVar = regularizationFactor + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel);
predictionError(1) = trainTarget(1);

toleranceDistanceSq = toleranceDistance^2;

% start training
for n = 2:trainSize
    
    % comparing the distance between trainInput(:,n) and the dictionary
    distance2dictionary = min(sum((trainInput(:,n)*ones(1,dictSize) - trainInput(:,dictionaryIndex)).^2));
    if (distance2dictionary < toleranceDistanceSq)
        if flagLearningCurve, learningCurve(n) = learningCurve(n-1); end
        continue;
    end
    
    %calc the Conditional Information
    k_vector = ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex),typeKernel,paramKernel);
    networkOutput = expansionCoefficient*k_vector;
    predictionError(n) = trainTarget(n) - networkOutput;
    
    if (abs(predictionError(n)) < tolerancePredictError)
        if flagLearningCurve==1, learningCurve(n) = learningCurve(n-1); end
            continue;
    end
    
    f_vector = Q_matrix*k_vector;
    
    skew(n) = -skewness(trainTarget);
%     m = median(predictionError);
%     s = std(predictionError);
%     v = min(0.99,abs(sum(predictionError-m/s)/n));
% %     v = abs(mean(predictionError)-median(predictionError)/std(predictionError));
%     d = sqrt((pi/2)*(v^(2/3))/(v^(2/3)+((4-pi)/2)^(2/3)));
%     skew(n) = real(d/sqrt(1-d^2));
    theta = 1/wt(predictionError(n),ks,skew(n));
    
    predictionVar = regularizationFactor*forgettingfactor^n*(ks^2)*theta +...
        ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) -...
        f_vector'*k_vector;
    
    %update Q_matrix
    s = 1/predictionVar;
    Q_tmp = zeros(dictSize+1,dictSize+1);
    Q_tmp(1:dictSize,1:dictSize) = Q_matrix + f_vector*f_vector'*s;
    Q_tmp(1:dictSize,dictSize+1) = -f_vector*s;
    Q_tmp(dictSize+1,1:dictSize) = Q_tmp(1:dictSize,dictSize+1)';
    Q_tmp(dictSize+1,dictSize+1) = s;
    Q_matrix = Q_tmp;
    
    % updating coefficients
    dictSize = dictSize + 1;
    disp(['KRMC-NC-Dictsize...',num2str(dictSize)])
    dictionaryIndex(dictSize) = n;
    expansionCoefficient(dictSize) = s*predictionError(n);
    expansionCoefficient(1:dictSize-1) = expansionCoefficient(1:dictSize-1) - (f_vector*s*predictionError(n))';

    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
             y_te(jj) = expansionCoefficient*...
                ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex),typeKernel,paramKernel);
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end

end

figure, plot(skew)

return
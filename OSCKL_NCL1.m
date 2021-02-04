function [expansionCoefficient,biasCoefficient,dictionaryIndex,learningCurve,identOutlier,estDict] = ...
    OSCKL_NCL1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,~,forgettingfactor,toleranceDistance,tolerancePredictError,regularization,flagLearningCurve,ks,epson_updt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Desenvolvido por Michael Santos Duarte
%PPGETI
%Set 20, 2018
%
%Descrição:
%Verificar os resultados do algoritimo OS-CKL - Online Sparse Correntropy Kernel Learning
%Recursivo
%NCL1 - Novelty Criterion with L1-norm regularization function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%=========OS-CKL===================

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
% wt = @(e,ks) exp(-(e).^2./(2*ks^2)) ./ (sqrt(2*pi)*ks^3); % Weighted terms

nc = @(e,ks) (1/(sqrt(2*pi)*ks)).*exp(-((e).^2)./(2*ks^2));
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

y_tr = zeros(trainSize,1);
y_tr(1) = expansionCoefficient'*ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel) - biasCoefficient(1);

% end level 2

% DICTIONARY
dictSize = 1;
dictionaryIndex(dictSize) = 1;
toleranceDistance = toleranceDistance^2;
estDict(1) = dictSize;

% test nonconvex regularization
%w(1) = 1;
%w2(1) = 1;

% end dictionary
dictSizeNC  = 1;
dictSizeP = 0;
adddict(1) = dictSizeNC;
remdict(1) = dictSizeP;
    
for i = 2:trainSize
    
    flag_continue = 0;

    % comparing the distance between trainInput(:,i) and the dictionary
    distance2dictionary = min(sum((trainInput(:,i)*ones(1,dictSize) - trainInput(:,dictionaryIndex)).^2));
    
    if (distance2dictionary < toleranceDistance)
    if flagLearningCurve, learningCurve(i) = learningCurve(i-1); end
        predictionError(i) = trainTarget(i) - expansionCoefficient'*ker_eval(trainInput(:,i),trainInput(:,dictionaryIndex),typeKernel,paramKernel) - biasCoefficient(1); % error of predicted values       
        aux_pe(i) = 1;
        pe(i) = wt(predictionError(i),ks); % new weighted terms
        flag_continue = 1;
        y_tr(i) = expansionCoefficient'*ker_eval(trainInput(:,i),trainInput(:,dictionaryIndex),typeKernel,paramKernel) - biasCoefficient(1);
    end
    predictionError(i) = trainTarget(i) - expansionCoefficient'*ker_eval(trainInput(:,i),trainInput(:,dictionaryIndex),typeKernel,paramKernel) - biasCoefficient(1); % error of predicted values       
    if (abs(predictionError(i)) < tolerancePredictError) %|| (dictSize > 500)
        if flagLearningCurve==1, learningCurve(i) = learningCurve(i-1); end
        aux_pe(i) = 1;
        pe(i) = wt(predictionError(i),ks); % new weighted terms
        flag_continue = 1;
        y_tr(i) = expansionCoefficient'*ker_eval(trainInput(:,i),trainInput(:,dictionaryIndex),typeKernel,paramKernel) - biasCoefficient(1);
        dictSizeP = dictSizeP + 0;
        dictSizeNC = dictSizeNC + 0;
    end
    
    % updating    
    if flag_continue == 0
        %Calc the Conditional Information    
        k_vector = ker_eval(trainInput(:,i),trainInput(:,dictionaryIndex),typeKernel,paramKernel);    

        %UPDATE MATRIX GRAM
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
        disp(['NCL1-Dictsize...',num2str(dictSize)])
        dictionaryIndex(dictSize) = i;    
        
        dictSizeNC = dictSizeNC + 1;
        adddict2(i) = 1;

        % LEVEL 1

        I = ones(dictSize,1); % vector ones

        auxTarget = trainTarget(dictionaryIndex);   

        % updating coefficients       
        expansionCoefficient = (P*(auxTarget - ((I*I'*P*auxTarget)./(I'*auxTarget*I))));% lagrange
        biasCoefficient = (I'*P*auxTarget)./(I'*auxTarget*I); % bias  
   
        % L1 regularization

        w = 1./(abs(expansionCoefficient)+0.1);
%         rho = 10;
%         sig = .1;
%         M = 1;
%         w = rho.*(1./(M.*sig.^2.*sqrt(2.*pi))).*expansionCoefficient.*exp(-(expansionCoefficient.^2)/(2.*sig.^2));
%         gm = 1;
%         w = max((1-abs(w)/gm*regularization),0); % MCP
        %w = .5*abs(w)^(.5-1); %lp, p<1
        %w = 1/(1+abs(w));%log
        expansionCoefficient = sign(expansionCoefficient).*max((abs(expansionCoefficient)-regularization.*w),0);
        w2 = 1./(abs(biasCoefficient)+0.1);
%         w2 = rho.*(1./(M.*sig.^2.*sqrt(2.*pi))).*biasCoefficient.*exp(-(biasCoefficient.^2)/(2.*sig.^2));
%         w2 = max((1-abs(w2)/gm*regularization),0);%MCP
        %w2 = .5*abs(w)^(.5-1); %lp, p<1
        %w2 = 1/(1+abs(w));%log
        biasCoefficient = sign(biasCoefficient).*max((abs(biasCoefficient)-regularization.*w2),0);

%       expansionCoefficient = sign(expansionCoefficient).*max((abs(expansionCoefficient)-regularization),0);

        sizej = dictSize;
        removeDic = [];
        for j = 1:sizej
            if (expansionCoefficient(j) == 0)  && (dictSize > 1)                      
                dictSize = dictSize - 1;
                disp(['NCL1-Dictsize...',num2str(dictSize)])
                removeDic = [removeDic j];    
            end
        end     
        
        dictSizeP = dictSizeP + length(removeDic);
        remdict2(i) = length(removeDic);

        if (sizej > dictSize)          
            for j = 1:length(removeDic)
                % Method paper Ruiz (Submatrix)
                K = []; Q = []; A = []; R =[]; e = []; v = []; Anew = []; u = [];

                for n = 1:sizej
                    K(n,1) = ker_eval(trainInput(:,dictionaryIndex(removeDic(j))),trainInput(:,dictionaryIndex(n)),typeKernel,paramKernel); % Kernel matrix
                end

                Q = zeros(sizej,1);
                Q(removeDic(j)) = 1./(regularizationFactor.*pe(dictionaryIndex(removeDic(j))));

                A = K + Q; % matrix symmetric positive-definite invertible

                R = zeros(sizej,1);  %Transform matrix to row canonical form (reduced row echelon form - RREF)
                R(removeDic(j)) = 1;

                e = R; 
                v = R'; 
                u = A - e;        

                Anew = P + ((P*u)*(v*P))/(1-v*P*u);
                Anew(:,removeDic(j)) = [];
                Anew(removeDic(j),:) = [];
                P = [];
                P = Anew; % inverse matrix

                dictionaryIndex(removeDic(j)) = [];
                expansionCoefficient(removeDic(j)) = [];
                biasCoefficient(removeDic(j)) = [];

                sizej = sizej - 1;
                removeDic = removeDic - 1;

            end             
 
%             I = ones(dictSize,1); % vector ones
% 
%             auxTarget = trainTarget(dictionaryIndex);   

%             % updating coefficients       
%             expansionCoefficient = (P*(auxTarget - ((I*I'*P*auxTarget)./(I'*auxTarget*I))));% lagrange
%             biasCoefficient = (I'*P*auxTarget)./(I'*auxTarget*I); % bias   

        end
        
        % end L1 regularization  
        
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
            else
                pe(dictionaryIndex(j)) = aux_pe(dictionaryIndex(j));
            end
        end
        
        I = ones(dictSize,1); % vector ones

        auxTarget = trainTarget(dictionaryIndex);   

        expansionCoefficient = P*(auxTarget - ((I*I'*P*auxTarget)./(I'*auxTarget*I)));% lagrange
        biasCoefficient = (I'*P*auxTarget)./(I'*auxTarget*I); % bias
        
        y_tr(i) = expansionCoefficient'*ker_eval(trainInput(:,i),trainInput(:,dictionaryIndex),typeKernel,paramKernel) - biasCoefficient(1);
        
        % TESTING OR VALIDATE
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

    estDict(i) = dictSize;
    adddict(i) = dictSizeNC;
    remdict(i) = dictSizeP;
    
end

identOutlier = pe;

figure,
% subplot(2,1,1)
plot(estDict,'LineWidth',2);
hold on
plot(adddict,'LineWidth',2)
plot(remdict,'LineWidth',2)
legend('Current','Added','Discarded','fontsize',14,'interpreter','latex','Location','best');
xlabel('time (steps)','fontsize',14,'interpreter','latex')
ylabel('number of items','fontsize',14,'interpreter','latex')
xlim([-inf inf])
ylim([-inf inf])
set(gca, 'XScale', 'log') 
set(gca, 'YScale', 'log') 
title('Evolution of the dictionary size');

figure,
plot(estDict);
hold on
% plot(adddict2)
stem(remdict2)
legend('Real','Discarded','LineWidth',4,'fontsize',14,'interpreter','latex','Location','best');
xlabel('time (steps)','fontsize',14,'interpreter','latex')
ylabel('samples','fontsize',14,'interpreter','latex')
ylim([-inf inf])
set(gca, 'YScale', 'log') 
title('Evolution of the size of dictionary');


% subplot(2,1,2)
figure,
hold on;
plot(trainTarget,'m--','LineWidth',1);
plot(y_tr,'r','LineWidth',1);
legend('Desired','Predicted');
axis([1 45000 -.2 .2]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers');
hold off;


%======end CKL===========

if flagLearningCurve
    figure;
    plot(learningCurve);
    set(gca, 'FontSize', 14);
    set(gca, 'FontName', 'Arial');
    xlabel('Step'),ylabel('Mean Squared Error (dB)');
    title('MSE (dB) one step ahead');
    
    % Animation
    figure;
    grid on;
    p = animatedline('Marker','.','MaximumNumPoints', length(estDict),'Color','b');
    axis([1 trainSize 1 max(estDict)])
    view(2);
    for i=1:length(estDict)
       addpoints(p,i,estDict(i));
       drawnow
    end    
end

return
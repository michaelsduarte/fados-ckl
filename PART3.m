%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Developed by Michael Santos Duarte
%PPGETI
%Set 17, 2018
%
%Description:
%Comparison of the results of the OS-CKL algorithm with different criteria of
%dictionary construction
%OS-CKL (Online Sparse Correntropy Kernel Learning)
%ALD (Approximate Linear Dependence)
%NC (Novelty Criterion)
%CC (Coherence Criterion)
%SC (Surprise Criterion)
%Algorithm type: Recursive
%Dataset: Wiener-Hammerstein
%One step ahead
%Multi step ahead
%Nonlinear Auoregressive Ex (NARX)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all, clear all
clc

%%

%======data formatting===========
% data settings
Nt  = 168000; % number of samples 188,000

load('WienerHammerBenchMark.mat')
 
uu=uBenchMark(10000:188000-10000)';  % select the section of the experiment
yy=yBenchMark(10000:188000-10000)';

%======data embedding=======
na = 10; % Order output
nb = 10; % Order input
d = 0; % Delay
nu = na+nb+1; % Parameters
x(:,1) = zeros(nu,1);
%data size
N_tr = 70000;
N_val = 48000;
N_te = 50000;%
%%======end of data=======

for i = 2:Nt
      
    for j=1:na
        if i-j <= 0
            x(j,i) = 0;
        else
            x(j,i) = -yy(i-j);
        end
    end
    for j=0:nb
        if i-j-d <= 0
            x(j+1+na,i) = 0;
        else
            x(j+1+na,i) = uu(i-j-d);
        end
    end
end

%======noise================
% pdf do ruído (Mixtura de ruídos impulsivos)
eps = [0 0.05];
t2 = N_tr; % Quantidade de amostras
m1 = -.1;
m2 = .2;
s1 = .06;
s2 = .06;
%s2 = var(yy)*3; %Desvio da gaussiana 2
Z = zeros(t2,length(eps)); %Inicializa variavel do ruído
nOutlier = length(eps);
% for no = 1:nOutlier
%     alpha = 1-eps(no); %Shape da mixtura de gaussiana
%     U = rand(t2,1); %Gera aleatoriamente vetor de base do ruido impulsivo
%     I1 = (U < alpha); %Define o shape do ruido
%     Z(:,no) = I1.*(randn(t2,1)*s1 + m1) + (1-I1).*(randn(t2,1)*s2 + m2); %Gera o ruido da mistura de gaussianas
% end
%%======end noise=======

% training data
trainInput = x(:,50001:120000);
trainTarget = yy(50001:120000)';

% validate data
valInput = x(:,120001:Nt);
valTarget = yy(120001:Nt)';

% testing data
testInput = x(:,1:N_te);
testTarget = yy(1:N_te)';
testTargetdefault = yy(1:N_te)';

% Data size for training and testing
trainSize = length(trainTarget);
valSize = length(valTarget);
testSize = length(testTarget);

%======end of data formatting===========

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%             SIMULATIONS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = .01;
nTrial = 1; % Number of simulations
%nOutlier = nOutlier;
%nOutlier = 1;
trainTarget0 = trainTarget;

for no = 1:nOutlier
    
    for nt = 1:nTrial
        
        alpha = 1-eps(no); %Shape da mixtura de gaussiana
        U = rand(t2,1); %Gera aleatoriamente vetor de base do ruido impulsivo
        I1 = (U < alpha); %Define o shape do ruido
        %Z(:,no) = I1.*0 + (1-I1).*((randn(t2,1)*s1 + m1) + (randn(t2,1)*s2 + m2)); %Gera o ruido da mistura de gaussianas
        %Z(:,no) = I1.*0 + (1-I1).*(var(yy)*trnd(2)); %Gera o ruido da mistura de t-student
        Z(:,no) = I1.*0 + (1-I1).*random('poiss', 0.05, t2, 1);

        trainTarget = trainTarget0+Z(:,no);
        disp(['Outlier # ',num2str(no.'),'/',num2str(nOutlier), '|| Trial # ',num2str(nt.'),'/',num2str(nTrial)]);
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %              OS-CKL-ALD
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Parameters
        regularizationFactor2 = 0.00000;  % Regularization parameter ald
        forgettingfactor = 1; % Forgetting factor
        flagLearningCurve = 0;
        length_ald = 1;
        %th_ald_vector = linspace(0.0055,0.0055,length_ald);
        if no == 1
            th_ald_vector = .0105;%.064;
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        else
            th_ald_vector = .0105;%.064;
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        end
        %th_ald_vector = .1;
        mse_cskl_ald = zeros(length_ald, 1);
        distsize_cskl_ald = zeros(length_ald, 1);

        fprintf('Learning OS-CKL-ALD...\n');

        for ii = 1:length_ald 
            
            disp(['Dictionary # ',num2str(ii.'),'/',num2str(length_ald)]);

            th_ald = th_ald_vector(ii);
            [expansionCoefficient,biasCoefficient,dictionaryIndex,learningCurve,identOutlier] = ...
                CKL_ALD(trainInput,trainTarget,valInput,valTarget,typeKernel,paramKernel,regularizationFactor,regularizationFactor2,forgettingfactor,th_ald,flagLearningCurve,ks,epson_updt);

            % ONE STEP AHEAD
            y_teCSKL = zeros(testSize,1);
            for jj = 1:testSize
                y_teCSKL(jj) = expansionCoefficient'*...
                    ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex),typeKernel,paramKernel) + biasCoefficient(1);
            end
            outlier(no).trial(nt).identOutlier(:,ii)  = identOutlier;
            outlier(no).trial(nt).distsize_cskl_ald(ii) = length(dictionaryIndex);
            outlier(no).trial(nt).mse_cskl_ald(ii) = mean((testTarget - y_teCSKL).^2);

            % MULT STEP AHEAD                
            y_kte_cskl = zeros(testSize,1);
            
            xkt(:,1) = zeros(nu,1);

            for jj = 1:testSize
                y_kte_cskl(jj) = expansionCoefficient'*ker_eval(xkt(:,jj),trainInput(:,dictionaryIndex),typeKernel,paramKernel) + biasCoefficient(1);

                yp(jj) = y_kte_cskl(jj);
                
                for j=1:na
                    if jj-j <= 0
                        xkt(j,jj+1) = 0;
                    else
                        xkt(j,jj+1) = -yp(jj+1-j);
                    end
                end
                for j=0:nb
                    if jj-j-d <= 0
                        xkt(j+1+na,jj+1) = 0;
                    else
                        xkt(j+1+na,jj+1) = uu(jj+1-j-d);
                    end
                end
            end

            outlier(no).trial(nt).rmse_cskl_msa(ii) = sqrt(mean((testTarget - y_kte_cskl).^2)); % RMSE

            error_cskl = testTarget - y_kte_cskl;
            outlier(no).trial(nt).error.msa.cskl(:,ii) = error_cskl;
            sig = 1;
            outlier(no).trial(nt).c_cskl(ii) = mean(exp(-((error_cskl).^2)./(2*sig^2))); % Correntropy

            outlier(no).trial(nt).output.osa.cskl(:,ii) = y_teCSKL;
            outlier(no).trial(nt).output.msa.cskl(:,ii) = y_kte_cskl;

        end

        % =========end of OS-CKL-ALD================
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %              OS-CKL-NC
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Parameters
        length_nc = 1;
        
        regularizationFactor2 = 0.00000;  % Regularization parameter ald
        forgettingfactor = 1; % Forgetting factor
        flagLearningCurve = 0;

        %th_ald_vector = linspace(0.0055,0.0055,length_ald);
        if no == 1
            th_distance_nc_vector = .6;%2; %.8 1,349
            th_error_nc_vector = .0206;
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        else
            th_distance_nc_vector = 1.2;%2.5; %.8
            th_error_nc_vector = .0206;
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        end
        %th_ald_vector = .1;
        mse_cskl_nc = zeros(length_nc, 1);
        distsize_cskl_nc = zeros(length_nc, 1);

        fprintf('Learning OS-CKL-NC...\n');

        for ii = 1:length_nc 
            
            disp(['Dictionary # ',num2str(ii.'),'/',num2str(length_ald)]);

            th_distance_nc = th_distance_nc_vector(ii);
                th_error_nc = th_error_nc_vector(ii);
                
            [expansionCoefficient2,biasCoefficient2,dictionaryIndex2,learningCurve2,identOutlier2] = ...
                OSCKL_NC(trainInput,trainTarget,valInput,valTarget,typeKernel,paramKernel,regularizationFactor,regularizationFactor2,forgettingfactor,th_distance_nc,th_error_nc_vector,flagLearningCurve,ks,epson_updt);

            % ONE STEP AHEAD
            y_teCSKLnc = zeros(testSize,1);
            for jj = 1:testSize
                y_teCSKLnc(jj) = expansionCoefficient2'*...
                    ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex2),typeKernel,paramKernel) + biasCoefficient2(1);
            end
            outlier(no).trial(nt).identOutliernc(:,ii)  = identOutlier2;
            outlier(no).trial(nt).distsize_cskl_nc(ii) = length(dictionaryIndex2);
            outlier(no).trial(nt).mse_cskl_nc(ii) = mean((testTarget - y_teCSKLnc).^2);

            % MULT STEP AHEAD                
            y_kte_csklnc = zeros(testSize,1);
            
            xkt(:,1) = zeros(nu,1);

            for jj = 1:testSize
                y_kte_csklnc(jj) = expansionCoefficient2'*ker_eval(xkt(:,jj),trainInput(:,dictionaryIndex2),typeKernel,paramKernel) + biasCoefficient2(1);

                yp(jj) = y_kte_csklnc(jj);
                
                for j=1:na
                    if jj-j <= 0
                        xkt(j,jj+1) = 0;
                    else
                        xkt(j,jj+1) = -yp(jj+1-j);
                    end
                end
                for j=0:nb
                    if jj-j-d <= 0
                        xkt(j+1+na,jj+1) = 0;
                    else
                        xkt(j+1+na,jj+1) = uu(jj+1-j-d);
                    end
                end
            end

            outlier(no).trial(nt).rmse_csklnc_msa(ii) = sqrt(mean((testTarget - y_kte_csklnc).^2)); % RMSE

            error_csklnc = testTarget - y_kte_csklnc;
            outlier(no).trial(nt).error.msa.csklnc(:,ii) = error_csklnc;
            sig = 1;
            outlier(no).trial(nt).c_csklnc(ii) = mean(exp(-((error_csklnc).^2)./(2*sig^2))); % Correntropy

            outlier(no).trial(nt).output.osa.csklnc(:,ii) = y_teCSKLnc;
            outlier(no).trial(nt).output.msa.csklnc(:,ii) = y_kte_csklnc;

        end

        % =========end of OS-CKL-NC================
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %              OS-CKL-CC
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Parameters
        length_cc = 1;
        
        regularizationFactor2 = 0.00000;  % Regularization parameter ald
        forgettingfactor = 1; % Forgetting factor
        flagLearningCurve = 0;

        %th_ald_vector = linspace(0.0055,0.0055,length_ald);
        if no == 1
            th_oscklcc = .9645;%.905; %.905
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        else
            th_oscklcc = .9645;%.905;
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        end
        %th_ald_vector = .1;
        mse_cskl_cc = zeros(length_cc, 1);
        distsize_cskl_cc = zeros(length_cc, 1);

        fprintf('Learning OS-CKL-CC...\n');

        for ii = 1:length_cc 
            
            disp(['Dictionary # ',num2str(ii.'),'/',num2str(length_ald)]);

            th_oscklcc_ii = th_oscklcc(ii);
                
            [expansionCoefficient3,biasCoefficient3,dictionaryIndex3,learningCurve3,identOutlier3] = ...
                OSCKL_CC(trainInput,trainTarget,valInput,valTarget,typeKernel,paramKernel,regularizationFactor,regularizationFactor2,forgettingfactor,th_oscklcc_ii,flagLearningCurve,ks,epson_updt);

            % ONE STEP AHEAD
            y_teCSKLcc = zeros(testSize,1);
            for jj = 1:testSize
                y_teCSKLcc(jj) = expansionCoefficient3'*...
                    ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex3),typeKernel,paramKernel) + biasCoefficient3(1);
            end
            outlier(no).trial(nt).identOutliercc(:,ii)  = identOutlier3;
            outlier(no).trial(nt).distsize_cskl_cc(ii) = length(dictionaryIndex3);
            outlier(no).trial(nt).mse_cskl_cc(ii) = mean((testTarget - y_teCSKLcc).^2);

            % MULT STEP AHEAD                
            y_kte_csklcc = zeros(testSize,1);
            
            xkt(:,1) = zeros(nu,1);

            for jj = 1:testSize
                y_kte_csklcc(jj) = expansionCoefficient3'*ker_eval(xkt(:,jj),trainInput(:,dictionaryIndex3),typeKernel,paramKernel) + biasCoefficient3(1);

                yp(jj) = y_kte_csklcc(jj);
                
                for j=1:na
                    if jj-j <= 0
                        xkt(j,jj+1) = 0;
                    else
                        xkt(j,jj+1) = -yp(jj+1-j);
                    end
                end
                for j=0:nb
                    if jj-j-d <= 0
                        xkt(j+1+na,jj+1) = 0;
                    else
                        xkt(j+1+na,jj+1) = uu(jj+1-j-d);
                    end
                end
            end

            outlier(no).trial(nt).rmse_csklcc_msa(ii) = sqrt(mean((testTarget - y_kte_csklcc).^2)); % RMSE

            error_csklcc = testTarget - y_kte_csklcc;
            outlier(no).trial(nt).error.msa.csklcc(:,ii) = error_csklcc;
            sig = 1;
            outlier(no).trial(nt).c_csklcc(ii) = mean(exp(-((error_csklcc).^2)./(2*sig^2))); % Correntropy

            outlier(no).trial(nt).output.osa.csklcc(:,ii) = y_teCSKLcc;
            outlier(no).trial(nt).output.msa.csklcc(:,ii) = y_kte_csklcc;

        end

        % =========end of OS-CKL-CC================
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %              OS-CKL-SC
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Parameters
        length_sc = 1;
        regularizationFactorSC = 0.0001;
        th1 = 1; 
        
        regularizationFactor2 = 0.00000;  % Regularization parameter ald
        forgettingfactor = 1; % Forgetting factor
        flagLearningCurve = 0;
        
        %th_ald_vector = linspace(0.0055,0.0055,length_ald);
        if no == 1
            th_oscklsc = -2.95;%-1.65 %-1.15;
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
            
        else
            th_oscklsc = -2.9;%-1.65 %-1.15;
            ks = .2; % Bandwidth correntropy
            epson_updt = .0001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        end
        %th_ald_vector = .1;
        mse_cskl_sc = zeros(length_sc, 1);
        distsize_cskl_sc = zeros(length_sc, 1);

        fprintf('Learning OS-CKL-SC...\n');

        for ii = 1:length_sc 
            
            disp(['Dictionary # ',num2str(ii.'),'/',num2str(length_sc)]);

            th_oscklsc_ii = th_oscklsc(ii);
                
            [expansionCoefficient4,biasCoefficient4,dictionaryIndex4,learningCurve4,identOutlier4] = ...
                OSCKL_SC(trainInput,trainTarget,valInput,valTarget,typeKernel,paramKernel,regularizationFactor,regularizationFactor2,forgettingfactor,th1,th_oscklsc_ii,regularizationFactorSC,flagLearningCurve,ks,epson_updt);

            % ONE STEP AHEAD
            y_teCSKLsc = zeros(testSize,1);
            for jj = 1:testSize
                y_teCSKLsc(jj) = expansionCoefficient4'*...
                    ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex4),typeKernel,paramKernel) + biasCoefficient4(1);
            end
            outlier(no).trial(nt).identOutliersc(:,ii)  = identOutlier4;
            outlier(no).trial(nt).distsize_cskl_sc(ii) = length(dictionaryIndex4);
            outlier(no).trial(nt).mse_cskl_sc(ii) = mean((testTarget - y_teCSKLsc).^2);

            % MULT STEP AHEAD                
            y_kte_csklsc = zeros(testSize,1);
            
            xkt(:,1) = zeros(nu,1);

            for jj = 1:testSize
                y_kte_csklsc(jj) = expansionCoefficient4'*ker_eval(xkt(:,jj),trainInput(:,dictionaryIndex4),typeKernel,paramKernel) + biasCoefficient4(1);

                yp(jj) = y_kte_csklsc(jj);
                
                for j=1:na
                    if jj-j <= 0
                        xkt(j,jj+1) = 0;
                    else
                        xkt(j,jj+1) = -yp(jj+1-j);
                    end
                end
                for j=0:nb
                    if jj-j-d <= 0
                        xkt(j+1+na,jj+1) = 0;
                    else
                        xkt(j+1+na,jj+1) = uu(jj+1-j-d);
                    end
                end
            end

            outlier(no).trial(nt).rmse_csklsc_msa(ii) = sqrt(mean((testTarget - y_kte_csklsc).^2)); % RMSE

            error_csklsc = testTarget - y_kte_csklsc;
            outlier(no).trial(nt).error.msa.csklsc(:,ii) = error_csklsc;
            sig = 1;
            outlier(no).trial(nt).c_csklsc(ii) = mean(exp(-((error_csklsc).^2)./(2*sig^2))); % Correntropy

            outlier(no).trial(nt).output.osa.csklsc(:,ii) = y_teCSKLsc;
            outlier(no).trial(nt).output.msa.csklsc(:,ii) = y_kte_csklsc;

        end

        % =========end of OS-CKL-SC================

    end % end nTrial
        
end % end nOutlier

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              RESULTS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%p_outlier = [0 5 10 15 20 25 30];
p_outlier = [0 5];
%p_outlier = [0];

for no = 1:nOutlier
    
    for nt = 1:nTrial
        best.cskl.index.outlier(no).trial(nt) = find(max(outlier(no).trial(nt).c_cskl)==outlier(no).trial(nt).c_cskl,1);
        best.cskl.osa.rmse.outlier(no).trial(nt) = sqrt(outlier(no).trial(nt).mse_cskl_ald(best.cskl.index.outlier(no).trial(nt)));
        best.cskl.msa.rmse.outlier(no).trial(nt) = outlier(no).trial(nt).rmse_cskl_msa(best.cskl.index.outlier(no).trial(nt));
        best.cskl.msa.ctp.outlier(no).trial(nt) = outlier(no).trial(nt).c_cskl(best.cskl.index.outlier(no).trial(nt));
        best.cskl.osa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.osa.cskl(:,best.cskl.index.outlier(no).trial(nt));
        best.cskl.msa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.msa.cskl(:,best.cskl.index.outlier(no).trial(nt));
        best.cskl.sizeDic.outlier(no).trial(nt) = outlier(no).trial(nt).distsize_cskl_ald(best.cskl.index.outlier(no).trial(nt));
        best.cskl.msa.output.identoutlier(no).trial(:,nt) = outlier(no).trial(nt).identOutlier(:,best.cskl.index.outlier(no).trial(nt));    
        best.cskl.msa.error.outlier(no).trial(:,nt) = outlier(no).trial(nt).error.msa.cskl(:,best.cskl.index.outlier(no).trial(nt));
        
        best.csklnc.index.outlier(no).trial(nt) = find(max(outlier(no).trial(nt).c_csklnc)==outlier(no).trial(nt).c_csklnc,1);
        best.csklnc.osa.rmse.outlier(no).trial(nt) = sqrt(outlier(no).trial(nt).mse_cskl_nc(best.csklnc.index.outlier(no).trial(nt)));
        best.csklnc.msa.rmse.outlier(no).trial(nt) = outlier(no).trial(nt).rmse_csklnc_msa(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.ctp.outlier(no).trial(nt) = outlier(no).trial(nt).c_csklnc(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.osa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.osa.csklnc(:,best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.msa.csklnc(:,best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.sizeDic.outlier(no).trial(nt) = outlier(no).trial(nt).distsize_cskl_nc(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.output.identoutlier(no).trial(:,nt) = outlier(no).trial(nt).identOutliernc(:,best.csklnc.index.outlier(no).trial(nt));    
        best.csklnc.msa.error.outlier(no).trial(:,nt) = outlier(no).trial(nt).error.msa.csklnc(:,best.csklnc.index.outlier(no).trial(nt));
        
        best.csklcc.index.outlier(no).trial(nt) = find(max(outlier(no).trial(nt).c_csklcc)==outlier(no).trial(nt).c_csklcc,1);
        best.csklcc.osa.rmse.outlier(no).trial(nt) = sqrt(outlier(no).trial(nt).mse_cskl_cc(best.csklcc.index.outlier(no).trial(nt)));
        best.csklcc.msa.rmse.outlier(no).trial(nt) = outlier(no).trial(nt).rmse_csklcc_msa(best.csklcc.index.outlier(no).trial(nt));
        best.csklcc.msa.ctp.outlier(no).trial(nt) = outlier(no).trial(nt).c_csklcc(best.csklcc.index.outlier(no).trial(nt));
        best.csklcc.osa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.osa.csklcc(:,best.csklcc.index.outlier(no).trial(nt));
        best.csklcc.msa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.msa.csklcc(:,best.csklcc.index.outlier(no).trial(nt));
        best.csklcc.sizeDic.outlier(no).trial(nt) = outlier(no).trial(nt).distsize_cskl_cc(best.csklcc.index.outlier(no).trial(nt));
        best.csklcc.msa.output.identoutlier(no).trial(:,nt) = outlier(no).trial(nt).identOutliercc(:,best.csklcc.index.outlier(no).trial(nt));    
        best.csklcc.msa.error.outlier(no).trial(:,nt) = outlier(no).trial(nt).error.msa.csklcc(:,best.csklcc.index.outlier(no).trial(nt));
        
        best.csklsc.index.outlier(no).trial(nt) = find(max(outlier(no).trial(nt).c_csklsc)==outlier(no).trial(nt).c_csklsc,1);
        best.csklsc.osa.rmse.outlier(no).trial(nt) = sqrt(outlier(no).trial(nt).mse_cskl_sc(best.csklsc.index.outlier(no).trial(nt)));
        best.csklsc.msa.rmse.outlier(no).trial(nt) = outlier(no).trial(nt).rmse_csklsc_msa(best.csklsc.index.outlier(no).trial(nt));
        best.csklsc.msa.ctp.outlier(no).trial(nt) = outlier(no).trial(nt).c_csklsc(best.csklsc.index.outlier(no).trial(nt));
        best.csklsc.osa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.osa.csklsc(:,best.csklsc.index.outlier(no).trial(nt));
        best.csklsc.msa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.msa.csklsc(:,best.csklsc.index.outlier(no).trial(nt));
        best.csklsc.sizeDic.outlier(no).trial(nt) = outlier(no).trial(nt).distsize_cskl_sc(best.csklsc.index.outlier(no).trial(nt));
        best.csklsc.msa.output.identoutlier(no).trial(:,nt) = outlier(no).trial(nt).identOutliersc(:,best.csklsc.index.outlier(no).trial(nt));    
        best.csklsc.msa.error.outlier(no).trial(:,nt) = outlier(no).trial(nt).error.msa.csklsc(:,best.csklsc.index.outlier(no).trial(nt));
        
%         best.csklsnc.index.outlier(no).trial(nt) = find(max(outlier(no).trial(nt).c_csklsnc)==outlier(no).trial(nt).c_csklsnc,1);
%         best.csklsnc.osa.rmse.outlier(no).trial(nt) = sqrt(outlier(no).trial(nt).mse_cskl_snc(best.csklsnc.index.outlier(no).trial(nt)));
%         best.csklsnc.msa.rmse.outlier(no).trial(nt) = outlier(no).trial(nt).rmse_csklsnc_msa(best.csklsnc.index.outlier(no).trial(nt));
%         best.csklsnc.msa.ctp.outlier(no).trial(nt) = outlier(no).trial(nt).c_csklsnc(best.csklsnc.index.outlier(no).trial(nt));
%         best.csklsnc.osa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.osa.csklsnc(:,best.csklsnc.index.outlier(no).trial(nt));
%         best.csklsnc.msa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.msa.csklsnc(:,best.csklsnc.index.outlier(no).trial(nt));
%         best.csklsnc.sizeDic.outlier(no).trial(nt) = outlier(no).trial(nt).distsize_cskl_snc(best.csklsnc.index.outlier(no).trial(nt));
%         best.csklsnc.msa.output.identoutlier(no).trial(:,nt) = outlier(no).trial(nt).identOutliersnc(:,best.csklsnc.index.outlier(no).trial(nt));    
%         best.csklsnc.msa.error.outlier(no).trial(:,nt) = outlier(no).trial(nt).error.msa.csklsnc(:,best.csklsnc.index.outlier(no).trial(nt));
    end
    
    best.cskl.osa.rmse.mean.outlier(no) = mean(best.cskl.osa.rmse.outlier(no).trial);
    best.cskl.msa.rmse.mean.outlier(no) = mean(best.cskl.msa.rmse.outlier(no).trial);
    best.cskl.msa.ctp.mean.outlier(no) = mean(best.cskl.msa.ctp.outlier(no).trial);
    best.cskl.osa.rmse.std.outlier(no) = std(best.cskl.osa.rmse.outlier(no).trial);
    best.cskl.msa.rmse.std.outlier(no) = std(best.cskl.msa.rmse.outlier(no).trial);
    best.cskl.msa.ctp.std.outlier(no) = std(best.cskl.msa.ctp.outlier(no).trial);
    best.cskl.index.best.outlier(no) = find(max(best.cskl.msa.ctp.outlier(no).trial)==best.cskl.msa.ctp.outlier(no).trial,1);
    best.cskl.osa.output.data.outlier(:,no) = best.cskl.osa.output.outlier(no).trial(:,best.cskl.index.best.outlier(no));
    best.cskl.msa.output.data.outlier(:,no) = best.cskl.msa.output.outlier(no).trial(:,best.cskl.index.best.outlier(no));
    best.cskl.msa.output.data.identoutlier(:,no) = best.cskl.msa.output.identoutlier(no).trial(:,best.cskl.index.best.outlier(no));
    best.cskl.sizeDic.min.outlier(no) = best.cskl.sizeDic.outlier(no).trial(best.cskl.index.best.outlier(no));
    best.cskl.msa.rmse.best.outlier(no) = best.cskl.msa.rmse.outlier(no).trial(best.cskl.index.best.outlier(no));
    best.cskl.sizeDic.mean.outlier(no) = mean(best.cskl.sizeDic.outlier(no).trial);
    best.cskl.sizeDic.std.outlier(no) = var(best.cskl.sizeDic.outlier(no).trial);   
    best.cskl.msa.error.data.outlier(:,no) = best.cskl.msa.error.outlier(no).trial(:,best.cskl.index.best.outlier(no));
    
    best.csklnc.osa.rmse.mean.outlier(no) = mean(best.csklnc.osa.rmse.outlier(no).trial);
    best.csklnc.msa.rmse.mean.outlier(no) = mean(best.csklnc.msa.rmse.outlier(no).trial);
    best.csklnc.msa.ctp.mean.outlier(no) = mean(best.csklnc.msa.ctp.outlier(no).trial);
    best.csklnc.osa.rmse.std.outlier(no) = std(best.csklnc.osa.rmse.outlier(no).trial);
    best.csklnc.msa.rmse.std.outlier(no) = std(best.csklnc.msa.rmse.outlier(no).trial);
    best.csklnc.msa.ctp.std.outlier(no) = std(best.csklnc.msa.ctp.outlier(no).trial);
    best.csklnc.index.best.outlier(no) = find(max(best.csklnc.msa.ctp.outlier(no).trial)==best.csklnc.msa.ctp.outlier(no).trial,1);
    best.csklnc.osa.output.data.outlier(:,no) = best.csklnc.osa.output.outlier(no).trial(:,best.csklnc.index.best.outlier(no));
    best.csklnc.msa.output.data.outlier(:,no) = best.csklnc.msa.output.outlier(no).trial(:,best.csklnc.index.best.outlier(no));
    best.csklnc.msa.output.data.identoutlier(:,no) = best.csklnc.msa.output.identoutlier(no).trial(:,best.csklnc.index.best.outlier(no));
    best.csklnc.sizeDic.min.outlier(no) = best.csklnc.sizeDic.outlier(no).trial(best.csklnc.index.best.outlier(no));
    best.csklnc.msa.rmse.best.outlier(no) = best.csklnc.msa.rmse.outlier(no).trial(best.csklnc.index.best.outlier(no));
    best.csklnc.sizeDic.mean.outlier(no) = mean(best.csklnc.sizeDic.outlier(no).trial);
    best.csklnc.sizeDic.std.outlier(no) = var(best.csklnc.sizeDic.outlier(no).trial);   
    best.csklnc.msa.error.data.outlier(:,no) = best.csklnc.msa.error.outlier(no).trial(:,best.csklnc.index.best.outlier(no));
    
    best.csklcc.osa.rmse.mean.outlier(no) = mean(best.csklcc.osa.rmse.outlier(no).trial);
    best.csklcc.msa.rmse.mean.outlier(no) = mean(best.csklcc.msa.rmse.outlier(no).trial);
    best.csklcc.msa.ctp.mean.outlier(no) = mean(best.csklcc.msa.ctp.outlier(no).trial);
    best.csklcc.osa.rmse.std.outlier(no) = std(best.csklcc.osa.rmse.outlier(no).trial);
    best.csklcc.msa.rmse.std.outlier(no) = std(best.csklcc.msa.rmse.outlier(no).trial);
    best.csklcc.msa.ctp.std.outlier(no) = std(best.csklcc.msa.ctp.outlier(no).trial);
    best.csklcc.index.best.outlier(no) = find(max(best.csklcc.msa.ctp.outlier(no).trial)==best.csklcc.msa.ctp.outlier(no).trial,1);
    best.csklcc.osa.output.data.outlier(:,no) = best.csklcc.osa.output.outlier(no).trial(:,best.csklcc.index.best.outlier(no));
    best.csklcc.msa.output.data.outlier(:,no) = best.csklcc.msa.output.outlier(no).trial(:,best.csklcc.index.best.outlier(no));
    best.csklcc.msa.output.data.identoutlier(:,no) = best.csklcc.msa.output.identoutlier(no).trial(:,best.csklcc.index.best.outlier(no));
    best.csklcc.sizeDic.min.outlier(no) = best.csklcc.sizeDic.outlier(no).trial(best.csklcc.index.best.outlier(no));
    best.csklcc.msa.rmse.best.outlier(no) = best.csklcc.msa.rmse.outlier(no).trial(best.csklcc.index.best.outlier(no));
    best.csklcc.sizeDic.mean.outlier(no) = mean(best.csklcc.sizeDic.outlier(no).trial);
    best.csklcc.sizeDic.std.outlier(no) = var(best.csklcc.sizeDic.outlier(no).trial);   
    best.csklcc.msa.error.data.outlier(:,no) = best.csklcc.msa.error.outlier(no).trial(:,best.csklcc.index.best.outlier(no));
    
    best.csklsc.osa.rmse.mean.outlier(no) = mean(best.csklsc.osa.rmse.outlier(no).trial);
    best.csklsc.msa.rmse.mean.outlier(no) = mean(best.csklsc.msa.rmse.outlier(no).trial);
    best.csklsc.msa.ctp.mean.outlier(no) = mean(best.csklsc.msa.ctp.outlier(no).trial);
    best.csklsc.osa.rmse.std.outlier(no) = std(best.csklsc.osa.rmse.outlier(no).trial);
    best.csklsc.msa.rmse.std.outlier(no) = std(best.csklsc.msa.rmse.outlier(no).trial);
    best.csklsc.msa.ctp.std.outlier(no) = std(best.csklsc.msa.ctp.outlier(no).trial);
    best.csklsc.index.best.outlier(no) = find(max(best.csklsc.msa.ctp.outlier(no).trial)==best.csklsc.msa.ctp.outlier(no).trial,1);
    best.csklsc.osa.output.data.outlier(:,no) = best.csklsc.osa.output.outlier(no).trial(:,best.csklsc.index.best.outlier(no));
    best.csklsc.msa.output.data.outlier(:,no) = best.csklsc.msa.output.outlier(no).trial(:,best.csklsc.index.best.outlier(no));
    best.csklsc.msa.output.data.identoutlier(:,no) = best.csklsc.msa.output.identoutlier(no).trial(:,best.csklsc.index.best.outlier(no));
    best.csklsc.sizeDic.min.outlier(no) = best.csklsc.sizeDic.outlier(no).trial(best.csklsc.index.best.outlier(no));
    best.csklsc.msa.rmse.best.outlier(no) = best.csklsc.msa.rmse.outlier(no).trial(best.csklsc.index.best.outlier(no));
    best.csklsc.sizeDic.mean.outlier(no) = mean(best.csklsc.sizeDic.outlier(no).trial);
    best.csklsc.sizeDic.std.outlier(no) = var(best.csklsc.sizeDic.outlier(no).trial);   
    best.csklsc.msa.error.data.outlier(:,no) = best.csklsc.msa.error.outlier(no).trial(:,best.csklsc.index.best.outlier(no));
    
%     best.csklsnc.osa.rmse.mean.outlier(no) = mean(best.csklsnc.osa.rmse.outlier(no).trial);
%     best.csklsnc.msa.rmse.mean.outlier(no) = mean(best.csklsnc.msa.rmse.outlier(no).trial);
%     best.csklsnc.msa.ctp.mean.outlier(no) = mean(best.csklsnc.msa.ctp.outlier(no).trial);
%     best.csklsnc.osa.rmse.std.outlier(no) = std(best.csklsnc.osa.rmse.outlier(no).trial);
%     best.csklsnc.msa.rmse.std.outlier(no) = std(best.csklsnc.msa.rmse.outlier(no).trial);
%     best.csklsnc.msa.ctp.std.outlier(no) = std(best.csklsnc.msa.ctp.outlier(no).trial);
%     best.csklsnc.index.best.outlier(no) = find(max(best.csklsnc.msa.ctp.outlier(no).trial)==best.csklsnc.msa.ctp.outlier(no).trial,1);
%     best.csklsnc.osa.output.data.outlier(:,no) = best.csklsnc.osa.output.outlier(no).trial(:,best.csklsnc.index.best.outlier(no));
%     best.csklsnc.msa.output.data.outlier(:,no) = best.csklsnc.msa.output.outlier(no).trial(:,best.csklsnc.index.best.outlier(no));
%     best.csklsnc.msa.output.data.identoutlier(:,no) = best.csklsnc.msa.output.identoutlier(no).trial(:,best.csklsnc.index.best.outlier(no));
%     best.csklsnc.sizeDic.min.outlier(no) = best.csklsnc.sizeDic.outlier(no).trial(best.csklsnc.index.best.outlier(no));
%     best.csklsnc.msa.rmse.best.outlier(no) = best.csklsnc.msa.rmse.outlier(no).trial(best.csklsnc.index.best.outlier(no));
%     best.csklsnc.sizeDic.mean.outlier(no) = mean(best.csklsnc.sizeDic.outlier(no).trial);
%     best.csklsnc.sizeDic.std.outlier(no) = var(best.csklsnc.sizeDic.outlier(no).trial);   
%     best.csklsnc.msa.error.data.outlier(:,no) = best.csklsnc.msa.error.outlier(no).trial(:,best.csklsnc.index.best.outlier(no));
end

% RMSE - ONE STEP AHEAD
figure
hold on
errorbar(p_outlier,best.cskl.osa.rmse.mean.outlier,best.cskl.osa.rmse.std.outlier, 'b-+','LineWidth',2)
errorbar(p_outlier,best.csklnc.osa.rmse.mean.outlier,best.csklnc.osa.rmse.std.outlier, 'r-+','LineWidth',2)
errorbar(p_outlier,best.csklcc.osa.rmse.mean.outlier,best.csklcc.osa.rmse.std.outlier, 'k-+','LineWidth',2)
errorbar(p_outlier,best.csklsc.osa.rmse.mean.outlier,best.csklsc.osa.rmse.std.outlier, 'g-+','LineWidth',2)
% errorbar(p_outlier,best.csklsnc.osa.rmse.mean.outlier,best.csklsnc.osa.rmse.std.outlier, 'm-+','LineWidth',2)
legend('OS-CKL-ALD','OS-CKL-NC','OS-CKL-CC','OS-CKL-SC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - One step ahead');

% RMSE - FREE SIMULATION
figure
hold on
errorbar(p_outlier,best.cskl.msa.rmse.mean.outlier,best.cskl.msa.rmse.std.outlier, 'b-+','LineWidth',2)
errorbar(p_outlier,best.csklnc.msa.rmse.mean.outlier,best.csklnc.msa.rmse.std.outlier, 'r-+','LineWidth',2)
errorbar(p_outlier,best.csklcc.msa.rmse.mean.outlier,best.csklcc.msa.rmse.std.outlier, 'k-+','LineWidth',2)
errorbar(p_outlier,best.csklsc.msa.rmse.mean.outlier,best.csklsc.msa.rmse.std.outlier, 'g-+','LineWidth',2)
% errorbar(p_outlier,best.csklsnc.msa.rmse.mean.outlier,best.csklsnc.msa.rmse.std.outlier, 'm-+','LineWidth',2)
legend('OS-CKL-ALD','OS-CKL-NC','OS-CKL-CC','OS-CKL-SC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
box on
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Errorbar');

figure
hold on
plot(p_outlier,best.cskl.msa.rmse.mean.outlier, 'b-+','LineWidth',2)
plot(p_outlier,best.csklnc.msa.rmse.mean.outlier, 'r-+','LineWidth',2)
plot(p_outlier,best.csklcc.msa.rmse.mean.outlier, 'k-+','LineWidth',2)
plot(p_outlier,best.csklsc.msa.rmse.mean.outlier, 'g-+','LineWidth',2)
% plot(p_outlier,best.csklsnc.msa.rmse.mean.outlier, 'm-+','LineWidth',2)
legend('OS-CKL-ALD','OS-CKL-NC','OS-CKL-CC','OS-CKL-SC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Plot');

figure
hold on
plot(p_outlier,best.cskl.msa.rmse.best.outlier, 'b-+','LineWidth',2)
plot(p_outlier,best.csklnc.msa.rmse.best.outlier, 'r-+','LineWidth',2)
plot(p_outlier,best.csklcc.msa.rmse.best.outlier, 'k-+','LineWidth',2)
plot(p_outlier,best.csklsc.msa.rmse.best.outlier, 'g-+','LineWidth',2)
% plot(p_outlier,best.csklsnc.msa.rmse.best.outlier, 'm-+','LineWidth',2)
legend('OS-CKL-ALD','OS-CKL-NC','OS-CKL-CC','OS-CKL-SC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Best');

% CORRENTROPY - FREE SIMULATION
figure
hold on
errorbar(p_outlier,best.cskl.msa.ctp.mean.outlier.*100,best.cskl.msa.ctp.std.outlier.*100, 'b-+','LineWidth',2)
errorbar(p_outlier,best.csklnc.msa.ctp.mean.outlier.*100,best.csklnc.msa.ctp.std.outlier.*100, 'r-+','LineWidth',2)
errorbar(p_outlier,best.csklcc.msa.ctp.mean.outlier.*100,best.csklcc.msa.ctp.std.outlier.*100, 'k-+','LineWidth',2)
errorbar(p_outlier,best.csklsc.msa.ctp.mean.outlier.*100,best.csklsc.msa.ctp.std.outlier.*100, 'g-+','LineWidth',2)
% errorbar(p_outlier,best.csklsnc.msa.ctp.mean.outlier.*100,best.csklsnc.msa.ctp.std.outlier.*100, 'm-+','LineWidth',2)
legend('OS-CKL-ALD','OS-CKL-NC','OS-CKL-CC','OS-CKL-SC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('% of correntropy');
title('CORRENTROPY - Free simulation - Errorbar');

figure
hold on
plot(p_outlier,best.cskl.msa.ctp.mean.outlier.*100, 'b-+','LineWidth',2)
plot(p_outlier,best.csklnc.msa.ctp.mean.outlier.*100, 'r-+','LineWidth',2)
plot(p_outlier,best.csklcc.msa.ctp.mean.outlier.*100, 'k-+','LineWidth',2)
plot(p_outlier,best.csklsc.msa.ctp.mean.outlier.*100, 'g-+','LineWidth',2)
% plot(p_outlier,best.csklsnc.msa.ctp.mean.outlier.*100, 'm-+','LineWidth',2)
legend('OS-CKL-ALD','OS-CKL-NC','OS-CKL-CC','OS-CKL-SC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('% of correntropy');
title('CORRENTROPY - Free simulation - Plot');

% DICTIONARY
figure
hold on
dic = [best.cskl.sizeDic.min.outlier; best.csklnc.sizeDic.min.outlier; best.csklcc.sizeDic.min.outlier; best.csklsc.sizeDic.min.outlier];
bar(p_outlier,dic')
legend('OS-CKL-ALD','OS-CKL-NC','OS-CKL-CC','OS-CKL-SC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('Size dictionary');
title('Min Dictionary of support vectors');

figure
hold on
dic2 = [best.cskl.sizeDic.mean.outlier; best.csklnc.sizeDic.mean.outlier; best.csklcc.sizeDic.mean.outlier; best.csklsc.sizeDic.mean.outlier];
bar(p_outlier,dic2')
legend('OS-CKL-ALD','OS-CKL-NC','OS-CKL-CC','OS-CKL-SC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('Size dictionary');
box on
title('Mean Dictionary of support vectors');

%best.csklsc.sizeDic.std.outlier(no)

% Testing OSA
% figure;
% plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
% hold on;
% plot(testTargetdefault,'m--','LineWidth',1);
% plot(y_teCSKL,'r','LineWidth',2);
% legend('Samples','Desired','Simulated');
% axis([0 40000 -.4 .4]);
% xlabel('time (steps)')
% ylabel('y')
% title('Predict output - One Step Ahead - OS-CKL');
% hold off;

% Testing MSA
figure;
subplot(5,2,1)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.cskl.msa.output.data.outlier(:,1),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output without outliers - OS-CKL-ALD');
hold off;

% Testing MSA
subplot(5,2,2)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.cskl.msa.output.data.outlier(:,nOutlier),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers - OS-CKL-ALD');
hold off;

subplot(5,2,3)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.output.data.outlier(:,1),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output without outliers - OS-CKL-NC');
hold off;

% Testing MSA
subplot(5,2,4)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.output.data.outlier(:,nOutlier),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers - OS-CKL-NC');
hold off;

subplot(5,2,5)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklcc.msa.output.data.outlier(:,1),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output without outliers - OS-CKL-CC');
hold off;

% Testing MSA
subplot(5,2,6)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklcc.msa.output.data.outlier(:,nOutlier),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers - OS-CKL-CC');
hold off;

subplot(5,2,7)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklsc.msa.output.data.outlier(:,1),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output without outliers - OS-CKL-SC');
hold off;

% Testing MSA
subplot(5,2,8)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklsc.msa.output.data.outlier(:,nOutlier),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 50000 -1.5 1.5]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers - OS-CKL-SC');
hold off;

% subplot(5,2,9)
% plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
% hold on;
% plot(testTargetdefault,'m--','LineWidth',1);
% plot(best.csklsnc.msa.output.data.outlier(:,1),'r','LineWidth',2);
% legend('Samples','Desired','Simulated');
% axis([0 50000 -1.5 1.5]);
% xlabel('time (steps)')
% ylabel('y')
% title('Predict output without outliers - OS-CKL-SNC');
% hold off;
% 
% % Testing MSA
% subplot(5,2,10)
% plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
% hold on;
% plot(testTargetdefault,'m--','LineWidth',1);
% plot(best.csklsnc.msa.output.data.outlier(:,nOutlier),'r','LineWidth',2);
% legend('Samples','Desired','Simulated');
% axis([0 50000 -1.5 1.5]);
% xlabel('time (steps)')
% ylabel('y')
% title('Predict output with 5% of outliers - OS-CKL-SNC');
% hold off;

% Error testing MSA
figure;
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.error.data.outlier(:,1) ,'r','LineWidth',2);
legend('Samples','Desired','Simulated');
xlabel('time (steps)')
ylabel('y')
title('Error predict output without outliers - OS-CKL-ALD');
hold off;

figure;
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.error.data.outlier(:,nOutlier) ,'r','LineWidth',2);
legend('Samples','Desired','Simulated');
xlabel('time (steps)')
ylabel('y')
title('Error predict output with 5% outliers - OS-CKL-ALD');
hold off;


%% Relation between outliers, correntropy and size dictionary
% 
% y1 = p_outlier';
% x1 = (100.-(best.cskl.sizeDic.mean.outlier./trainSize.*100))';
% z1 = (best.cskl.msa.ctp.mean.outlier.*100)';
% name = 'OS-CKL-ALD';
% plot_mesh2(x1,y1,z1);
% 
% color1 = [0 0 1];    % make plot symbols blue 
% sf1 = 50;
% legwords1 = 'CSKL';
% %-------------------------------------------------
% %close all
% line(x1(1),y1(1),'Color',color1,'Visible','off');  % plot only the first point from each dataset so can use
% hold on                                            % legend
% 
% h1 = bubbleplot(x1,y1,z1,color1,sf1);  % plot first dataset
% hold on;                                      % note hold must be turned back 'on' after each call to bubbleplot 
% 
% set(gca,'FontSize',8);
% xlabel('% Sparsity');
% ylabel('% Outliers');
% legend(legwords1,'Location','Best','Orientation','horizontal');
% box on
% % set(gcf,'Color',[.8 .8 .8],'InvertHardCopy','off');
% 
% 
% %% Identify outliers
% 
% % Normalized weights
% for no = 1:nOutlier
%     best.cskl.msa.output.data.identoutlier(:,nOutlier)  = ((best.cskl.msa.output.data.identoutlier(:,nOutlier)  -...
%         min(best.cskl.msa.output.data.identoutlier(:,nOutlier) )) /...
%         ( max(best.cskl.msa.output.data.identoutlier(:,nOutlier) ) -...
%         min(best.cskl.msa.output.data.identoutlier(:,nOutlier) ) )); % Normalizado [0 1]
% end
% cutoff = 0.95*ones(length(best.cskl.msa.output.data.identoutlier(:,nOutlier) ),1);
% 
% figure;
% plot(best.cskl.msa.output.data.identoutlier(:,nOutlier));
% %plot(best.cskl.msa.output.data.identoutlier(:,nOutlier) );
% hold on
% plot(cutoff,'--');
% axis([0, N_tr, 0, 1]) 
% title('Normalized weights (pe)');
% 
% xdata = (1:trainSize)';
% I = best.cskl.msa.output.data.identoutlier(:,nOutlier)  <= .95 ; 
% outliersDetect = excludedata(xdata,trainTarget,'indices',I);
% 
% I = abs(trainTarget0 - trainTarget) > .1; 
% outliersTrue = excludedata(xdata,trainTarget,'indices',I);
% 
% for i=1:trainSize
%     if outliersDetect(i)==0 && outliersTrue(i)==0 % Corretamente classificado como não outlier
%         TN(i) = 1;
%     else
%         TN(i) = 0;
%     end
%     if outliersDetect(i)==1 && outliersTrue(i)==1 % Corretamente classificado como outlier
%         TP(i) = 1;
%     else
%         TP(i) = 0;
%     end
%     if outliersDetect(i)==0 && outliersTrue(i)==1 % Outlier esquecido
%         FN(i) = 1;
%     else
%         FN(i) = 0;
%     end
%     if outliersDetect(i)==1 && outliersTrue(i)==0 % Outlier classificado incorretamente
%         FP(i) = 1;
%     else
%         FP(i) = 0;
%     end
% end
% 
% figure;
% plot(xdata(find((TN==1))),trainTarget(find((TN==1))),'b.','MarkerSize',15)
% hold on
% plot(xdata(find((TP==1))),trainTarget(find((TP==1))),'r.','MarkerSize',15)
% plot(xdata(find((FN==1))),trainTarget(find((FN==1))),'g.','MarkerSize',15)
% plot(xdata(find((FP==1))),trainTarget(find((FP==1))),'m.','MarkerSize',15)
% legend('TN','TP','FN','FP');
% xlabel('time (steps)')
% ylabel('y')
% title('Ouliers detection');

for i = 1:nTrial
    sd1(i) = outlier(2).trial(i).distsize_cskl_ald;
    sd2(i) = outlier(2).trial(i).distsize_cskl_nc;
    sd3(i) = outlier(2).trial(i).distsize_cskl_sc;
    sd4(i) = outlier(2).trial(i).distsize_cskl_cc;
end
std(sd1)
std(sd2)
std(sd3)
std(sd4)

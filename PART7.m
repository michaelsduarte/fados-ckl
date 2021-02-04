%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Developed by Michael Santos Duarte
%PPGETI
%Out 12, 2018
%
%Description:
% Comparison of the results of the OS-CKL algorithm with the novelty criterion and
% pruning
%OS-CKL (Online Sparse Correntropy Kernel Learning)
%NCL1 (Novelty Criterion with [or adaptive] L1-norm regularization function)
%Algorithm type: Recursive
%Dataset: Silverbox
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
Nt  = 131072; % number of samples 131,072

load('SNLS80mV.mat')
 
uSNL=V1(1:Nt);  % select the section of the experiment
ySNL=V2(1:Nt);

%======data embedding=======
na = 10; % Order output
nb = 10; % Order input
d = 0; % Delay
nu = na+nb+1; % Parameters
x(:,1) = zeros(nu,1);
%data size
N_tr = 45000;
N_val = 46072;
N_te = 40000;%
%%======end of data=======

for i = 2:Nt
      
    for j=1:na
        if i-j <= 0
            x(j,i) = 0;
        else
            x(j,i) = -ySNL(i-j);
        end
    end
    for j=0:nb
        if i-j-d <= 0
            x(j+1+na,i) = 0;
        else
            x(j+1+na,i) = uSNL(i-j-d);
        end
    end
end

%======noise================
% pdf do ruído (Mixtura de ruídos impulsivos)
eps = [0 0.05];
t2 = Nt; % Quantidade de amostras
m1 = 0; % Media da gaussiana 1
m2 = 0; % Media da gausssiana 2
s1 = 0.01; %Desvio da gaussiana 1
s2 = var(ySNL)*30; %Desvio da gaussiana 2
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
trainInput = x(:,40001:85000);
trainTarget = ySNL(40001:85000)';

% validate data
valInput = x(:,85001:Nt);
valTarget = ySNL(85001:Nt)';

% testing data
testInput = x(:,1:N_te);
testTarget = ySNL(1:N_te)';
testTargetdefault = ySNL(1:N_te)';

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
typeKernel = 'Poly';
paramKernel = 3;
nTrial = 1; % Number of simulations
trainTarget0 = trainTarget;

for no = 1:nOutlier
    
    for nt = 1:nTrial
        
        alpha = 1-eps(no); %Shape da mixtura de gaussiana
        U = rand(t2,1); %Gera aleatoriamente vetor de base do ruido impulsivo
        I1 = (U < alpha); %Define o shape do ruido
%         Z(:,no) = I1.*0 + (1-I1).*(randn(t2,1)*s2 + m2); %Gera o ruido da mistura de gaussianas
         Z(:,no) = I1.*0 + (1-I1).*(var(ySNL)*trnd(2)); %Gera o ruido da mistura de t-student
        
%         alpha = 1-eps(no); %Shape da mixtura de gaussiana
%         U = rand(t2,1); %Gera aleatoriamente vetor de base do ruido impulsivo
%         I1 = (U < alpha); %Define o shape do ruido
%         
%         Z(:,no) = Z(:,no) + I1.*0 + (1-I1).*(2*var(ySNL)*trnd(2));

        trainTarget = trainTarget0+Z(40001:85000,no);
        
%         figure, plot(trainTarget), hold on, plot(trainTarget0)

        disp(['Outlier # ',num2str(no.'),'/',num2str(nOutlier), '|| Trial # ',num2str(nt.'),'/',num2str(nTrial)]);
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %              OS-CKL-NCL1
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Parameters
        length_nc = 1;
        
        regularizationFactor2 = 0.00000;  % Regularization parameter ald
        forgettingfactor = 1; % Forgetting factor
        flagLearningCurve = 0;

        %th_ald_vector = linspace(0.0055,0.0055,length_ald);
        if no == 1
            th_distance_nc_vector = linspace(0.15,0.15,length_nc); %.15 % 2.8936 % 6.42 %.23
            th_error_nc_vector  = linspace(0.001,0.001,length_nc);
            ks = .2; % Bandwidth correntropy
            epson_updt = .00001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        else
            th_distance_nc_vector = linspace(0.15,0.15,length_nc); % .215
            th_error_nc_vector  = linspace(0.001,0.001,length_nc);
            ks = .4; % Bandwidth correntropy
            epson_updt = .00001; % Criterion update matrix
            regularizationFactor = 10; % Regularization parameter correntropy
        end
        mse_cskl_nc = zeros(length_nc, 1);
        distsize_cskl_nc = zeros(length_nc, 1);

        fprintf('Learning OS-CKL-NCL1...\n');

        for ii = 1:length_nc 
            
            disp(['Dictionary # ',num2str(ii.'),'/',num2str(length_nc)]);

            th_distance_nc = th_distance_nc_vector(ii);
                th_error_nc = th_error_nc_vector(ii);
                
            [expansionCoefficient2,biasCoefficient2,dictionaryIndex2,learningCurve2,identOutlier2] = ...
                OSCKL_NCL1(trainInput,trainTarget,valInput,valTarget,typeKernel,paramKernel,regularizationFactor,regularizationFactor2,forgettingfactor,th_distance_nc,th_error_nc_vector,0.005,flagLearningCurve,ks,epson_updt);
            
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
                        xkt(j+1+na,jj+1) = uSNL(jj+1-j-d);
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

        % =========end of OS-CKL-NCL1================
        
    outlier(no).trial(nt).norma = norm(expansionCoefficient2);
        
    end % end nTrial
        
end % end nOutlier

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              RESULTS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p_outlier = [0 5];

for no = 1:nOutlier
    
    for nt = 1:nTrial
        
        best.csklnc.index.outlier(no).trial(nt) = find(max(outlier(no).trial(nt).c_csklnc)==outlier(no).trial(nt).c_csklnc,1);
        best.csklnc.osa.rmse.outlier(no).trial(nt) = sqrt(outlier(no).trial(nt).mse_cskl_nc(best.csklnc.index.outlier(no).trial(nt)));
        best.csklnc.msa.rmse.outlier(no).trial(nt) = outlier(no).trial(nt).rmse_csklnc_msa(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.ctp.outlier(no).trial(nt) = outlier(no).trial(nt).c_csklnc(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.osa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.osa.csklnc(:,best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.output.outlier(no).trial(:,nt) = outlier(no).trial(nt).output.msa.csklnc(:,best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.sizeDic.outlier(no).trial(nt) = outlier(no).trial(nt).distsize_cskl_nc(best.csklnc.index.outlier(no).trial(nt));
        best.csklnc.msa.output.identoutlier(no).trial(:,nt) = outlier(no).trial(nt).identOutliernc(:,best.csklnc.index.outlier(no).trial(nt));    
        best.csklnc.msa.error.outlier(no).trial(:,nt) = outlier(no).trial(nt).error.msa.csklnc(:,best.csklnc.index.outlier(no).trial(nt));
    end
   
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
    best.csklnc.norma.outlier(no) =  outlier(no).trial(best.csklnc.index.best.outlier(no)).norma;  
    
end

% Norma alfa
best.csklnc.norma.outlier

% RMSE - ONE STEP AHEAD
figure
hold on
errorbar(p_outlier,best.csklnc.osa.rmse.mean.outlier,best.csklnc.osa.rmse.std.outlier, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - One step ahead');

% RMSE - FREE SIMULATION
figure
hold on
errorbar(p_outlier,best.csklnc.msa.rmse.mean.outlier,best.csklnc.msa.rmse.std.outlier, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
box on
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Errorbar');

figure
hold on
plot(p_outlier,best.csklnc.msa.rmse.mean.outlier, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Plot');

figure
hold on
plot(p_outlier,best.csklnc.msa.rmse.best.outlier, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('RMSE');
title('RMSE - Free simulation - Best');

% CORRENTROPY - FREE SIMULATION
figure
hold on
errorbar(p_outlier,best.csklnc.msa.ctp.mean.outlier.*100,best.csklnc.msa.ctp.std.outlier.*100, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('% of correntropy');
title('CORRENTROPY - Free simulation - Errorbar');

figure
hold on
plot(p_outlier,best.csklnc.msa.ctp.mean.outlier.*100, 'r-+','LineWidth',2)
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('% of correntropy');
title('CORRENTROPY - Free simulation - Plot');

% DICTIONARY
figure
hold on
dic = [best.csklnc.sizeDic.min.outlier];
bar(p_outlier,dic')
legend('OS-CKL-NC');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('Size dictionary');
title('Min Dictionary of support vectors');

figure
hold on
dic2 = [best.csklnc.sizeDic.mean.outlier];
bar(p_outlier,dic2')
legend('OS-CKL-NCL1');
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('% of outlier'),ylabel('Size dictionary');
box on
title('Mean Dictionary of support vectors');

% Testing MSA
figure;
subplot(2,1,1)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.output.data.outlier(:,1),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 40000 -.4 .4]);
xlabel('time (steps)')
ylabel('y')
% title('Predict output without outliers - OS-CKL-NCL1');
hold off;

% Testing MSA
subplot(2,1,2)
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.output.data.outlier(:,nOutlier),'r','LineWidth',2);
legend('Samples','Desired','Simulated');
axis([0 40000 -.4 .4]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers - OS-CKL-NCL1');
hold off;

figure;
subplot(2,1,1)
range = 32000:32250;
plot(range,testTarget(32000:32250),'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(range,testTargetdefault(32000:32250),'m--','LineWidth',1);
plot(range, best.csklnc.msa.output.data.outlier(32000:32250,1),'r','LineWidth',2);
% legend('Samples','Desired','Simulated');
axis([32000 32250 -.2 .2]);
xlabel('time (steps)')
ylabel('y')
title('Predict output without outliers');
hold off;

subplot(2,1,2)
range = 32000:32250;
plot(range,testTarget(32000:32250),'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(range,testTargetdefault(32000:32250),'m--','LineWidth',1);
plot(range, best.csklnc.msa.output.data.outlier(32000:32250,2),'r','LineWidth',2);
% legend('Samples','Desired','Simulated');
axis([32000 32250 -.2 .2]);
xlabel('time (steps)')
ylabel('y')
title('Predict output with 5% of outliers');
hold off;


% Error testing MSA
figure;
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.error.data.outlier(:,1) ,'r','LineWidth',2);
legend('Samples','Desired','Simulated');
xlabel('time (steps)')
ylabel('y')
title('Error predict output without outliers - OS-CKL-NCL1');
hold off;

figure;
plot(testTarget,'b.','MarkerSize',15,'LineWidth',1);
hold on;
plot(testTargetdefault,'m--','LineWidth',1);
plot(best.csklnc.msa.error.data.outlier(:,nOutlier) ,'r','LineWidth',2);
legend('Samples','Desired','Simulated');
xlabel('time (steps)')
ylabel('y')
% title('Error predict output with 5% outliers - OS-CKL-NCL1');
hold off;


%%
% figure,
% range = 32001:32250;
% 
% plot(range(1),testTarget(32001),'b.','MarkerSize',15,'LineWidth',1);
% hold on;
% plot(range,testTargetdefault(32001),'m--','LineWidth',1);
% % plot(range, best.csklnc.msa.output.data.outlier(32001,2),'r','LineWidth',2);
% % legend('Samples','Desired','Simulated');
% axis([32000 32250 -.2 .2]);
% xlabel('time (steps)')
% ylabel('y')
% title('Predict output with 5% of outliers');
% hold off;

% for i = 2:250
%     plot(range, best.csklnc.msa.output.data.outlier(32000+i,2),'r','LineWidth',2);
%     pause(.1)
% end
% hold off;

% %%
% h = figure
% 
% grid on;
% range = 32000:32250;
% p = animatedline('Marker','.','MaximumNumPoints', 251,'Color','b');
% p2 = animatedline('Marker','.','MaximumNumPoints', 251,'Color','r');
% axis([32000 32250 -.2 .2])
% view(2);
% legend('Desired','Simulated');
% 
% for i=1:length(range)
%    addpoints(p,range(i),testTarget(31999+i));
%    addpoints(p2,range(i),best.csklnc.msa.output.data.outlier(31999+i,2));
%    drawnow
% %    pause(.01)
%    
%       % Capture the plot as an image 
%       frame = getframe(h); 
%       im = frame2im(frame); 
%       [imind,cm] = rgb2ind(im,256); 
%       % Write to the GIF File 
%       if i == 1 
%           imwrite(imind,cm,'result.gif','gif', 'Loopcount',inf); 
%       else 
%           imwrite(imind,cm,'result.gif','gif','WriteMode','append'); 
%       end 
% end

for i = 1:nTrial
    sd1(i) = outlier(2).trial(i).distsize_cskl_nc(ii);
end
std(sd1)

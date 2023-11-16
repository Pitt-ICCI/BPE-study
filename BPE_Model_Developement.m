% This code was created by Dooman Arefan, University of PIttsburgh, ICCI
% lab
% Developement of prediction models for BPE measures from one or both
% breasts, tumor-derived radiomic features, or combinations of these, using
% LASSO-LDA for binary classification and LASSO-Logistic Regression for
% continous Oncotype DX score prediction

close all, clear all 
clc;

high_th=25;
k=10;

% Load computed BPE measures using an in-house validated software
[m, BPE_side3]=xlsread('path to the Ipsilateral BPE measures using cut-off=20%');
[m BPE_side4]=xlsread('path to the Ipsilateral BPE measures using cut-off=30%');
[m BPE_side5]=xlsread('path to the Ipsilateral BPE measures using cut-off=40%');

[m BPE_cont3]=xlsread('path to the contralateral BPE measures using cut-off=20%');
[m BPE_cont4]=xlsread('path to the contralateral BPE measures using cut-off=30%');
[m BPE_cont5]=xlsread('path to the contralateral BPE measures using cut-off=40%');

BPE_side3_t=csv2table(BPE_side3);
BPE_side4_t=csv2table(BPE_side4);
BPE_side5_t=csv2table(BPE_side5);

BPE_cont3_t=csv2table(BPE_cont3);
BPE_cont4_t=csv2table(BPE_cont4);
BPE_cont5_t=csv2table(BPE_cont5);

% Load tumor-derived radiomics features computed by Pyradiomics python code
xx=xlsread('path to the excel sheet contaning radiomics features for each patient/tumor');
[r c]=find(isnan(xx));
[yy,PS] = removerows(xx,'ind',r);

%Load patient's data including Oncotype DX score
onco=xlsread('path to the Excel sheet containing patients information');

for j=1:size(yy)
    ad=find(onco(:,1)==yy(j,1));
    onco_s(j,1)=onco(ad(1),18);
end

onco_s_b=onco_s>high_th;


x=([yy(:, 1:end) onco_s]);
Y_d_Nor=x;


cc=1;
for i=1:size(BPE_side3_t,1)
    r_ad=find(Y_d_Nor(:,1)==BPE_side3_t(i,1));
    if size(r_ad)>0

% Select one feature set as input for the model from the below options:

% Only Ipsilateral BPE measures
%Y_BPE(cc,:)=[Y_d_Nor(r_ad,1) BPE_side3_t(i,2:end) BPE_side4_t(i,2:end) BPE_side5_t(i,2:end) Y_d_Nor(r_ad,end)]; 

% Only Contralateral BPE measures
%Y_BPE(cc,:)=[Y_d_Nor(r_ad,1) BPE_cont3_t(i,2:end) BPE_cont4_t(i,2:end) BPE_cont5_t(i,2:end) Y_d_Nor(r_ad,end)]; 

%Combination of Ipsilateral and Contralateral BPE measures
%Y_BPE(cc,:)=[Y_d_Nor(r_ad,1) BPE_side3_t(i,2:end) BPE_side4_t(i,2:end) BPE_side5_t(i,2:end) BPE_cont3_t(i,2:end) BPE_cont4_t(i,2:end) BPE_cont5_t(i,2:end) Y_d_Nor(r_ad,end)];

% Only Radiomics features
%Y_BPE(cc,:)=[Y_d_Nor(r_ad,1) Y_d_Nor(r_ad,2:end-1) Y_d_Nor(r_ad,end)]; 

% Combination of Radiomics features and Ipsilateral BPE measures
%Y_BPE(cc,:)=[Y_d_Nor(r_ad,1) BPE_side3_t(i,2:end) BPE_side4_t(i,2:end) BPE_side5_t(i,2:end) Y_d_Nor(r_ad,2:end-1) Y_d_Nor(r_ad,end)]; 

% Combination of Radiomics features and Contralateral BPE measures
%Y_BPE(cc,:)=[Y_d_Nor(r_ad,1) BPE_cont3_t(i,2:end) BPE_cont4_t(i,2:end) BPE_cont5_t(i,2:end) Y_d_Nor(r_ad,2:end-1) Y_d_Nor(r_ad,end)]; 

% Combination of Radiomics features and Ipsilateral and Contralateral BPE measures
Y_BPE(cc,:)=[Y_d_Nor(r_ad,1) BPE_side3_t(i,2:end) BPE_side4_t(i,2:end) BPE_side5_t(i,2:end) BPE_cont3_t(i,2:end) BPE_cont4_t(i,2:end) BPE_cont5_t(i,2:end) Y_d_Nor(r_ad,2:end-1) Y_d_Nor(r_ad,end)]; 
cc=cc+1;
    end
end


% Normalize BPE measures using Min/Max method
for j=2:size(Y_BPE,2)-1
Y_BPE(:,j)=(Y_BPE(:,j)-min(Y_BPE(:,j)))/(max(Y_BPE(:,j))-min(Y_BPE(:,j)));
end
[rr cc]=find(isnan(Y_BPE));
Y_d_Nor=Y_BPE;
Y_d_Nor(:,cc)=[];


c_inx=Y_d_Nor(:,end);
class1_inx=find(c_inx==0); % Low/Intermediate risk Oncotype DX category
class2_inx=find(c_inx==1); % High risk Oncotype DX category

class1=(removerows(Y_d_Nor,'ind',[class2_inx]));
class2=(removerows(Y_d_Nor,'ind',[class1_inx]));

all_class_p=[class1;class2];

%Generate stratified k-fold cross validation seeds to split data into
%train/test sets
rng(1); % for reproducibility
cvFolds1 = crossvalind('Kfold', size(class1,1), k);
cvFolds2 = crossvalind('Kfold', size(class2,1), k);


%Loop for K-fold cross validation procedure
for fold=1:k
i=fold
TesInx1=(cvFolds1==i);
TesInx2=(cvFolds2==i);
TesInx=[TesInx1;TesInx2];

TrainInx= ~TesInx ;
trInd=find(TrainInx);
tesInd=find(TesInx);

Y_d_Nor_train=all_class_p(trInd,:);
Y_d_Nor_test=all_class_p(tesInd,:);
X_test=Y_d_Nor_test(:,2:end-1);
X_train=Y_d_Nor_train(:,2:end-1);
Y_test=Y_d_Nor_test(:,end);
Y_train=Y_d_Nor_train(:,end);

% Apply 5-fold cross-validated LASSO feature selection technique on the Training set
%[B,FitInfo] = lasso(X_train,Y_train,'cv',5);
[B,FitInfo] =lassoglm(X_train,Y_train,'binomial','cv',5);

C=(abs(B)>0);
CCC=sum(C);

lasso_t_f=FitInfo.IndexMinMSE;

r=(find(C(:,lasso_t_f)==1))' ;
feature_v=r;
num_features=size(feature_v,2);

% Selected Feature set based on min MSE in LASSO technique
X_train_lasso=X_train(:,feature_v);
X_test_lasso=X_test(:,feature_v);

% LDA model for the classification task
model = fitcdiscr(X_train_lasso,Y_train,'DiscrimType','linear');  % LDA
[predict_label, predict_score] = predict(model, X_test_lasso);

% Logistic Regression model to predict contnious Oncotype DX scores
% b = glmfit(X_train_lasso,Y_train_continous);
% Y_predicted_LR=[X_test_lasso(size(X_test_lasso,1),1)]*b;


% Compute Model Performance metrics
[Xroc_fold,Yroc_t,Troc,AUC(fold,:)] = perfcurve(Y_test,predict_score(:,2),1,'NBoot',1,'XVals',[0:0.05:1]);

thresh= 0.25; %[0:0.01:1];
predict_label=double(predict_score(:,2)> thresh);
Accuracy(fold)=sum(predict_label==Y_test)/size(Y_test,1);


% To calculate confusion matrix
y_es_test_cm=full(ind2vec(abs(predict_label'+1))); 
[c,cm,ind,per]=confusion(y_ac_test_cm,y_es_test_cm);
Sens(fold)=cm(2,2)/(cm(2,1)+cm(2,2));
Spec(fold)=cm(1,1)/(cm(1,1)+cm(1,2));
end


AUC_ave=mean(AUC(:,1))
acc_ave=mean(Accuracy)
Sen_ave=mean(Sens)
Spec_ave=mean(Spec)





function table_f=csv2table(csv_f) 
BPE_side=csv_f;
for i=1:size(BPE_side,1)
    BPE_side_str=cell2mat(BPE_side(i,1));
    ad_coma=find(BPE_side_str==','); 
    BPE_side_t(i,1)=str2num(BPE_side_str(1:ad_coma(1)-1));
    BPE_side_t(i,2)=str2num(BPE_side_str(ad_coma(1)+1:ad_coma(2)-1));
    BPE_side_t(i,3)=str2num(BPE_side_str(ad_coma(2)+1:ad_coma(3)-1));
    BPE_side_t(i,4)=str2num(BPE_side_str(ad_coma(3)+1:ad_coma(4)-1));
    BPE_side_t(i,5)=str2num(BPE_side_str(ad_coma(4)+1:ad_coma(5)-1));
    BPE_side_t(i,6)=str2num(BPE_side_str(ad_coma(5)+1:end-1));
end
table_f=BPE_side_t;
end




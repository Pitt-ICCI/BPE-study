% This code was created by Dooman Arefan, University of PIttsburgh, ICCI
% lab
% Developement of prediction models for BPE measures from one or both
% breasts, tumor-derived radiomic features, or combinations of these, using
% LASSO-LDA for binary classification and LASSO-Logistic Regression for
% continous Oncotype DX score prediction

close all, clear all 
clc;

high_th=25;


%Load patient's data including Oncotype DX score and menopausal status
onco=xlsread('path to the Excel sheet containing patients information');
meno=xlsread('path to the Excel sheet containing patients menopausal status');

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


for j=1:size(yy)
    ad=find(onco(:,1)==yy(j,1));
    onco_s(j,1)=onco(ad(1),18);
    ts(j,1)=onco(ad(1),6);
    age(j,1)=onco(ad(1),8);

end

onco_s_b=onco_s>high_th;
x=([yy(:, 1:end) onco_s_b]);
Y_d_Nor=x;

cc=0;
for i=1:size(BPE_side3_t,1)
    r_ad=find(Y_d_Nor(:,1)==BPE_side3_t(i,1));
    if size(r_ad)>0
     cc=cc+1;   
% Demonstrating Absolute (number of pixels) and relative (percentage) BPE measures
BPE_s3=[BPE_side3_t(i,2:3) BPE_side3_t(i,4:end)/100];
BPE_s4=[BPE_side4_t(i,2:3) BPE_side4_t(i,4:end)/100];
BPE_s5=[BPE_side5_t(i,2:3) BPE_side5_t(i,4:end)/100]; 
BPE_c3=[BPE_cont3_t(i,2:3) BPE_cont3_t(i,4:end)/100];
BPE_c4=[BPE_cont4_t(i,2:3) BPE_cont4_t(i,4:end)/100];
BPE_c5=[BPE_cont5_t(i,2:3) BPE_cont5_t(i,4:end)/100];

   Y_BPE(cc,:)=[Y_d_Nor(r_ad,1) BPE_side3_t(i,2:end) BPE_side4_t(i,2:end) BPE_side5_t(i,2:end)  BPE_cont3_t(i,2:end) BPE_cont4_t(i,2:end) BPE_cont5_t(i,2:end) Y_d_Nor(r_ad,end)]; 

   
    end
     ad2=find(meno(:,1)==BPE_side3_t(i,1));
   if size(ad2,1)>0
      meno2(cc,1)=meno(ad2(1),2);
   end
end


Y_d=Y_BPE;
for j=2:size(Y_BPE,2)-1
Y_BPE(:,j)=(Y_BPE(:,j)-min(Y_BPE(:,j)))/(max(Y_BPE(:,j))-min(Y_BPE(:,j)));
end

Y_d_Nor=Y_BPE;

% ***** Enable this for subgroup analysis of Menopausal
  % Y_d_Nor=Y_d_Nor(meno2<3,:);
% ****** end of subgroup analysis

Y=categorical(Y_d_Nor(:,end)>high_th);
for j=1:30

X=Y_d_Nor(:,j+1);
X_low_int=X(Y=="false");
X_high=X(Y=="true");

% Two-sample t-test to compare each BPE measure between high vs
% low/intermediate categories
[hhh p_val_ttest(j,1)]=ttest2(X_low_int,X_high);

% Univariable Logistic Regression 
mdl = fitglm(X,Y,'Distribution','binomial','Link','logit');

% Compute P-value and Odds ratio from logistic regression
p_val(j,1)=table2array(mdl.Coefficients(2,4));
Odd_r(j,1)=round(exp(table2array(mdl.Coefficients(2,1))),2);

% Compute 95% CI for Odds ratio
CI_95(j,1)=round(exp(table2array(mdl.Coefficients(2,1))-table2array(mdl.Coefficients(2,2))*1.96),2);
CI_95(j,2)=round(exp(table2array(mdl.Coefficients(2,1))+table2array(mdl.Coefficients(2,2))*1.96),2);
% 95% CI



end


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





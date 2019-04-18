    clc
clear
close all
x1=xlsread('1.xlsx');
x2=xlsread('2.xlsx');
x1(:,116)=0;
x2(:,116)=1;
X(1:499,:)=x1;
X(500:998,:)=x2;
[m n]=size(X);
save X X;
j=0;
for i=1:2:m
    j=j+1;
    XX(j,:)=X(i,:);
end
for i=2:2:m
    j=j+1;
    XX(j,:)=X(i,:);
end
X=XX;
[coeff,score,latent,tsquared,explained] = pca(X);
%%%%%%%%%%
%%%%%%%%%%
Train_Input=score(1:30,1:36);
T=Train_Input;
Train_Target=score(1:30,37);
C=Train_Target';

Test_Input=score(31:38,1:36);
test=Test_Input;
Test_Target=score(31:38,37);
itrind=size(test,1);
itrfin=[];
Cb=C;
Tb=T;
for tempind=1:itrind
    tst=test(tempind,:);
    C=Cb;
    T=Tb;
    u=unique(C);
    N=length(u);
    c4=[];
    c3=[];
    j=1;
    k=1;
    if(N>1)
        itr=1;
        classes=0;
        cond=max(C)-min(C);
        while((classes~=1)&&(itr<=length(u))&& size(C,2)>1 && cond>0)
            c1=(C==u(itr));
            newClass=c1;
            svmStruct = svmtrain(T,newClass,'kernel_function','rbf');  
            classes = svmclassify(svmStruct,tst);
        
            for i=1:size(newClass,2)
                if newClass(1,i)==0;
                    c3(k,:)=T(i,:);
                    k=k+1;
                end
            end
        T=c3;
        c3=[];
        k=1;
        
            for i=1:size(newClass,2)
                if newClass(1,i)==0;
                    c4(1,j)=C(1,i);
                    j=j+1;
                end
            end
        C=c4;
        c4=[];
        j=1;
        
        cond=max(C)-min(C); 
       
            if classes~=1
                itr=itr+1;
            end    
        end
    end

valt=Cb==u(itr);		
val=Cb(valt==1);		
val=unique(val);
itrfin(tempind,:)=val;  
end
[m n]=size(Test_Target);
sum=0;
for i=1:m
    if (Test_Target(i,1)~=itrfin(i,1))
        sum=sum+1;
    end
end

%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%
% save X XX;
tc = fitctree(X(1:38,1:115),X(1:38,116));
%%%%%%%%%%%%%%%%%%%%%%
load X2
attributes = {'a','b','c','d'};
description = 'Botnet Dataset';
[ds, uc, nf] = build_dataset(meas,species,attributes,description);

%% Prepare test and training sets. 
[train_dataset, test_dataset] = splitting_dataset(ds,0.7);

%% Run Naive Bayes
[train_targets_i, train_targets_l]=grp2idx(train_dataset.(5)); % Change class name into ordinal index
[test_targets_i, test_targets_l]=grp2idx(test_dataset.(5)); % Change class name into ordinal index

predicted_features = naive_bayes(double(train_dataset(:,1:4)), train_targets_i, double(test_dataset(:,1:4)));
%%%%%%%%%%%%%%%%%%%%%%%%%
LSTM_2();
%%%%%%%%%%%%%%%%%%%%
z=Train_Input;
y=Train_Target;
max_y=max(max(y));
y=y./max_y;
z=z./max(max(z));
h_n=60;
w1=rand(37,h_n);
w2=rand(h_n+1,1);
% w1=w1-.5;
% w2=w2-.5;
train_rate=0.3;
epoch=1000;
 z(:,37)=1;
 for j=1:epoch
     j
    for k=1:30
        input= z(k,:);
        target=y(k,:);
        first_out=tansig(input*w1);
        first_out(:,h_n+1)=1;
        final_out=tansig(first_out*w2);
        if final_out~=target
            first_delta=(target-final_out).*(train_rate*(1+final_out).*(1-final_out));
            second_delta=(first_delta*w2').*(train_rate.*(1+first_out).*(1-first_out));
            w2=w2+train_rate*(first_delta'*first_out )';
            [u,l]=size(second_delta);
            for ii=1:l-1
                h(1,ii)=second_delta(1,ii);
            end
            w1=w1+train_rate*(input'*h);
        end
    
       out=final_out;
    end
      for k=1:30
        input= z(k,:);
        target=y(k,:);
        first_out=tansig(input*w1);
        first_out(:,h_n+1)=1;
        final_out=tansig(first_out*w2);
        if final_out~=target
            first_delta=(target-final_out).*(train_rate*(1+final_out).*(1-final_out));
            second_delta=(first_delta*w2').*(train_rate.*(1+first_out).*(1-first_out));
            w2=w2+train_rate*(first_delta'*first_out )';
            [u,l]=size(second_delta);
            for ii=1:l-1
                h(1,ii)=second_delta(1,ii);
            end
            w1=w1+train_rate*(input'*h);
        end
    
       out=final_out;
    end
 end
%%%%%%%%%%%%%%%%%%%%
plot(x1,y1,'g','linewidth',2)
xlabel('Iteration')
ylabel('Error Rate')
hold on
plot(x1,y2,'r','linewidth',2)
hold on
plot(x1,y3,'linewidth',2)
legend('n=60','n=20','n=40'); 
load tbl
figure;
c = categorical({'F-Measure(%)','FP Rate(%)','TP Rate(%)'});
bar(c,tbl')
legend('Proposed Algorithm', 'BN','SVM', 'DT')
% nnn.m for learning a set of features from another set
% modified with the symmetric sigmoid (hyperbolic tangent) in the hidden layer
% and data normalisation to [-1,1] interval

%--------------preparing input and output data ----------------------
da=load('Data\iris.dat');
[n,m]=size(da);
     %-------normalise to [-10,10] scale----------------------
mr=max(da);
ml=min(da);
ra=mr-ml;
ba=mr+ml;
tda=2*da-ones(n,1)*ba;
dan=tda./(ones(n,1)*ra);
dan=10*dan;
%--------------preparing input and output-----------------
ip=[1:2];
ic=length(ip);
op=[3:4];
oc=length(op);
output=dan(:,op); %target file
input=dan(:,ip);  %input features
input(:,ic+1)=10;       %bias component
%--------initialising the network ----------------------------
h=40; %the number of hidden neurons
W=randn(ic+1,h) %initialising w weights
V=randn(h,oc) %initialising v weights
W0=W;
V0=V;
count=0; %counter of epochs
stopp=0; %stop-condition to change
%pause(3);

while(stopp==0)
mede=zeros(1,oc); % mean errors after an epoch
%----------cycling over entities in a random order
    ror=randperm(n);
    for ii=1:n
        x=input(ror(ii),:); %current instance's input
        u=output(ror(ii),:);% current instance's output    
%-----------forward pass (to calculate response ru)------
        ow=x*W;
        o1=1+exp(-ow);
        oow=ones(1,h)./o1;
        oow=2*oow-1;% symmetric sigmoid output of the hidden layer
        ov=oow*V; %output of the output layer
        err=u-ov; %the error
        mede=mede+abs(err)/n;
%---------error back-propagation--------------------------
        gV=-oow'*err;       % gradients of matrix V
        t1=V*err'; % error propagated to the hidden layer
        t2=(1-oow).*(1+oow)/2; %the derivative
        t3=t2.*t1';% error multiplied by the th's derivative
        gW=-x'*t3;         % gradients of matrix W
%-----------------change of the weights-----------------------
        mu=0.0001;
        V=V-mu*gV;
        W=W-mu*gW;
    end;
%------------------stop-condition --------------------------
    count=count+1;
    ss=mean(mede);
    if ss<0.01|count>=10000
        stopp=1;
    end;
    mede;
    if rem(count,1000)==0
        count
        mede
    end
end;
V
W
 
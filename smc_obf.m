function [psf_m, psf_sa]=smc_obf(xc,pf,S,prior)


%%%%%Optimal Bayesian feature filtering for structured multiclass data

%%%%%Please cite:
%%%%%Biomarker Discovery via Optimal Bayesian Feature Filtering for
%%%%%Structured Multiclass Data
%%%%%By Ali Foroughi pour and Lori A. dalton


%%%%%input:

%%%%%xc: is a 1 by c cell where c is the number of classes. Each element of
%%%%%xc is a nc by TF matrix where nc is the number of observations
%%%%%(points) in class c and TF is the total number of features

%%%%%pf: it is the prior probability that a feature f is a marker. If pf is
%%%%%a number then its value is used for all features. If pf is a 1 by TF
%%%%%vector then ith element of pf is the prior probability that the ith
%%%%%feature is a marker

%%%%%S: it is the set of possible strucures. It is a s by c matrix, where s
%%%%%is the number of possible structures and c is the number of classes.
%%%%%The first column is always all zero. S is comprisez of zeros and ones.
%%%%%zeros denote that a marker with that structure follows a distribution
%%%%%similar to the first class (first column). 1 denotes otherwise, i.e.,
%%%%%the distribution is not similar to class 0.


%%%%prior denotes the prior for each feature and the prior on S. If no
%%%%prior is inputted the code uses Jeffreys non-informative prior on
%%%%distribution paramters, and assumes all structures are equally likely.
%%%%To input a proper prior specify the following values:

%%%%sg0: the scale value of the inverse Wishart prior on the varince of a
%%%%marker in pattern 0 classes
%%%%kg0: The degrees of freedom of the inverse Wishart prior on the varince of a
%%%%marker in pattern 0 classes

%%%%NOTE the average variance of a marker in class 0 given sg0 and kg0 is
%%%%sg0/(kg0-2). Also, the larger kg0 is themore confidence that that variance of
%%%%a marker in class 0 is indeed sg0. kg0=3 is a good value, and hence, sg0 is the average
%%%%variance of a marker in class0. This should help to specify sg0 and kg0

%%%%sg1: the scale value of the inverse Wishart prior on the varince of a
%%%%marker in pattern 1 classes
%%%%kg1: The degrees of freedom of the inverse Wishart prior on the varince of a
%%%%marker in pattern 1 classes

%%%%sb: the scale value of the inverse Wishart prior on the varince of a
%%%%non-marker in all classes
%%%%kb: The degrees of freedom of the inverse Wishart prior on the varince of a
%%%%non-marker in all classes

%%%%mg0: The average mean of a marker in pattern 0 classes
%%%%nug0: the confidence on mg0. We assume given the variance in pattern
%%%% 0 classes (vg0), the prior on the mean of a feature is Gaussian with mean mg0 and
%%%%variance vg0/nug0.

%%%%mg1: The average mean of a marker in pattern 1 classes
%%%%nug1: the confidence on mg1. We assume given the variance in pattern
%%%% 1 classes (vg1), the prior on the mean is Gaussian with mean mg1 and
%%%%variance vg1/nug1.


%%%%mb: The average mean of a non-marker in all classes
%%%%nub: the confidence on mb. We assume given the variance in all classes
%%%% (vb), the prior on the mean is Gaussian with mean mb and
%%%%variance vb/nub.


%%%priS: is the prior vector on each Structure (row of S).


%%%%ouput:
%%%%psf_m: the marginal poterior probability that feature f is a marker
%%%%psf_sa: a  s+1 by TF matrix. Recall s=size(S,1). The first s rows are the posterior
%%%%probability that f is a marker with structures in S. The last row is
%%%%the posterior probability that f is a non-marker.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%construct the prior


if nargin<4
    
    sg0=0;
    sg1=0;
    sb=0;
    kg0=0;
    kg1=0;
    kb=0;
    mg0=0;
    mg1=0;
    mb=0;
    nug0=0;
    nug1=0;
    nub=0;
    
    priS=1/size(S,1);

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%weight of Jeffreys non-informative prior
    if size(S,1)<51
        jpw=0.05;
    else
        jpw=0.5;
    end
    
    
    input_pri=0;
    
    
else
    
    sg0=prior.sg0;
    sg1=prior.sg1;
    sb=prior.sb;
    kg0=prior.kg0;
    kg1=prior.kg1;
    kb=prior.kb;
    mg0=prior.mg0;
    mg1=prior.mg1;
    mb=prior.mb;
    nug0=prior.nug0;
    nug1=prior.nug1;
    nub=prior.nub;
    
    input_pri=1;
    
    priS=prior.priS;
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nclass=length(xc);

n_vec=zeros(1,nclass);

for i=1:nclass
    
    cx=xc{i};
    
    n_vec(i)=size(cx,1);
    
end

TF=size(cx,2);

pnum=size(S,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lpg=zeros(pnum+1,TF);



for pcnt=1:pnum
    
    cpat=S(pcnt,:);
    
    f0=find(cpat==0);
    f1=find(cpat==1);
    
    f0=f0(:)';
    f1=f1(:)';
    
    cn0=sum(n_vec(f0));
    cn1=sum(n_vec(f1));
    
    x0=zeros(cn0,TF);
    x1=zeros(cn1,TF);
    
    csn0=[0 cumsum(n_vec(f0))];
    csn1=[0 cumsum(n_vec(f1))];
    
    
    for i=1:length(f0)
        
        x0(1+csn0(i):csn0(i+1),:)=xc{f0(i)};
        
    end
    
    for i=1:length(f1)
        
        x1(1+csn1(i):csn1(i+1),:)=xc{f1(i)};
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    v0=var(x0);
    v1=var(x1);
    m0=mean(x0);
    m1=mean(x1);
    
    mgq0=(cn0*nug0)/(nug0+cn0);
    mgq1=(cn1*nug1)/(nug1+cn1);
    
    
    if input_pri>0
        
        s0s=sg0/(cn0-1)+v0+mgq0*(mg0-m0).^2/(cn0-1);
        s1s=sg1/(cn1-1)+v1+mgq1*(mg1-m1).^2/(cn1-1);
        
        q0=0.5*kg0*log(sg0)-gammaln(0.5*kg0)+0.5*log(nug0/(nug0+cn0))+gammaln(0.5*(kg0+cn0)) - 0.5*(kg0+cn0)*log(cn0-1);
        q1=0.5*kg1*log(sg1)-gammaln(0.5*kg1)+0.5*log(nug1/(nug1+cn1))+gammaln(0.5*(kg1+cn1)) - 0.5*(kg1+cn1)*log(cn1-1);
        
        lpg(pcnt,:)=q0+q1-0.5*((cn0+kg0)*log(s0s)+(cn1+kg1)*log(s1s))+log(priS)+log(pf);
        
        
    else
        
        
        q0=-0.5*log(cn0)+gammaln(0.5*cn0) - 0.5*cn0*log(cn0-1);
        q1=-0.5*log(cn1)+gammaln(0.5*cn1) - 0.5*cn1*log(cn1-1);
        
        lpg(pcnt,:)=q0+q1-0.5*(cn0*log(v0)+cn1*log(v1))+log(jpw)+log(priS)+log(pf);
        
        
    end
    
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Now the part we assume a feature is bad

nt=sum(n_vec);
csnt=[0 cumsum(n_vec)];

xb=zeros(nt,TF);

for i=1:nclass
    xb(1+csnt(i):csnt(i+1),:)=xc{i};
end

vb=var(xb);
mt=mean(xb);


mbq=(nt*nub)/(nub+nt);

if input_pri>0
    
    sbs=sbc/(nt-1)+vb+mbq*(mt-mb).^2/(nt-1);
    
    qt=0.5*kb*log(sb)-gammaln(0.5*kb)+0.5*log(nub/(nub+nt))+gammaln(0.5*(kb+nt)) - 0.5*(kb+nt)*log(nt-1);
    
    lpg(end,:)=qt-0.5*((nt+kb)*log(sbs))+log(1-pf);
    
else
    
    
    qt=-0.5*log(nt)+gammaln(0.5*nt) - 0.5*nt*log(nt-1);
    
    lpg(end,:)=qt-0.5*(nt*log(vb))+log(jpw)+log(1-pf);
    
    
end




slpg=lpg-repmat(lpg(end,:),pnum+1,1);

pg=exp(slpg);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


psf_sa=pg./repmat(sum(pg),pnum+1,1);


psf_m=1-psf_sa(end,:);











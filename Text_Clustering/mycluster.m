function [ class ] = mycluster( bow, K )
%
% Your goal of this assignment is implementing your own text clustering algo.
%
% Input:
%     bow: data set. Bag of words representation of text document as
%     described in the assignment.
%
%     K: the number of desired topics/clusters. 
%
% Output:
%     class: the assignment of each topic. The
%     assignment should be 1, 2, 3, etc. 
%
% For submission, you need to code your own implementation without using
% any existing libraries

% YOUR IMPLEMENTATION SHOULD START HERE!

%Initialize
doc_size=size(bow,1);
word_size=size(bow,2);
PI=ones(1,K)./K;
temp=rand(word_size,K);
mu=temp./repmat(sum(temp,1),[word_size,1]);
gamma=ones(doc_size,K);
error=1;
tol=10^(-6);
while error>tol
    %Expectation
    gamma1=gamma;
    for l=1:1:K
        gamma_temp(:,l)=PI(l).*prod(repmat(mu(:,l)',[doc_size,1]).^bow(:,:),2);
    end 
    sum_gamma_temp=sum(gamma_temp,2);
    gamma=gamma_temp./repmat(sum_gamma_temp,[1,K]);
    %Maximization
    mu_temp=bow'*gamma;
    mu=mu_temp./(repmat(sum(mu_temp,1),[word_size,1]));
    PI=sum(gamma,1)./doc_size;
    %
    error=norm(gamma-gamma1);    
end
[value,class]=max(gamma,[],2);
end


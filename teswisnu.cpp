#include <stdio.h>
#include <iostream>
using namespace std;
#define inf 999
#define n 5
#define mem 1
#define nonm 0

int main(){
	int bobot[n][n]={0,12,5,999,17,1,0,999,15,999,5,999,0,999,10,999,15,999,0,6,17,999,10,6,0};
	int s,t,pre[n];
	int dist[n],perm[n];
	int cur,i,k,dc,z;
	int smal,newd;
	cout<<"simpul asal(0-4):";
	cin>>s;
	cout<<"simpul tujuan (0-4):";
	cin>>t;
	
	for(i=0;i<=4;i++){
		perm[i]=nonm;
		dist[i]=inf;
	}
	
	perm[s]=mem;
	dist[s]=0;
	cur=s;
	while(cur!=t){
		smal=inf;
		dc=dist[cur];
		for(i=0;i<=4;i++){
			if(perm[i]==nonm){
				(newd=dc+bobot[cur][i]);
				if(newd<dist[i]){
					dist[i]=newd;
					pre[i]=cur;
				}
				if(dist[i]<smal){
					smal=dist[i];
					k=i;
				}
			}
		}
	cur=k;
	perm[cur]=mem;
	}
cout<<endl<<endl;
cout<<"graphnya adalah "<<dist[t];
}

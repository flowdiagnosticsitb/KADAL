//This is the implementation of variants of the 5term and slice-update scheme in which
//a vector of Gaussian PDFs is used instead of just one.
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>
#include "ehvi_hvol.h"
#include "ehvi_multi.h"
#include <iostream>
#include "helper.h"
#include "ehvi_consts.h"
#include <math.h>
double gausscdf1[1001][1001];
double gausscdf2[1001][1001];
double gausscdf3[1001][1001];
char flag1[1001][1001];
char flag2[1001][1001];
char flag3[1001][1001];
double exipsi1[1001][1001];
double exipsi2[1001][1001];
double exipsi3[1001][1001];
char fpsi1[1001][1001];
char fpsi2[1001][1001];
char fpsi3[1001][1001];




using namespace std;

vector<double> ehvi3d_5term(deque<shared_ptr<individual>> P, double r[], vector<shared_ptr<mus>> & pdf){
// vector<double> ehvi3d_5term(deque<individual*> P, double r[], vector<mus *> & pdf){
//5-term 3-dimensional ehvi calculation scheme. Subtracts 4 quantities off a rectangular volume.
  vector<double> answer; //The eventual answer
  int n = P.size(); //Holds amount of points.
  double Sminus; //Correction term for the integral.
  deque<shared_ptr<individual>> Py, Pz; //P sorted by y/z coordinate
  sort(P.begin(), P.end(), ycomparator);
  for (int i=0;i<P.size();i++){
    Py.push_back(P[i]);
  }
  sort(P.begin(), P.end(), zcomparator);
  for (unsigned int i=0;i<P.size();i++){
    Pz.push_back(P[i]);
  }
  sort(P.begin(), P.end(), xcomparator);
  for (int i=0;i<pdf.size();i++){
    answer.push_back(0);
  }
  for (int z=0;z<=n;z++){
    for (int y=0;y<=n;y++){
      for (int x=0;x<=n;x++){
        double v[DIMENSIONS]; //upper corner of hypervolume improvement box
        double cl[DIMENSIONS], cu[DIMENSIONS]; //Boundaries of grid cells
        cl[0] = (x == 0 ? r[0] : P[x-1]->f[0]);
        cl[1] = (y == 0 ? r[1] : Py[y-1]->f[1]);
        cl[2] = (z == 0 ? r[2] : Pz[z-1]->f[2]);
        cu[0] = (x == n ? INFINITY : P[x]->f[0]);
        cu[1] = (y == n ? INFINITY : Py[y]->f[1]);
        cu[2] = (z == n ? INFINITY : Pz[z]->f[2]);
        //We have to find v. This naive implementation is O(n) per iteration.
        v[0] = r[0];
        v[1] = r[1];
        v[2] = r[2];
        bool dominated = false;
        for (unsigned int i=0;i<P.size();i++){
          if (P[i]->f[0] >= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] >= cu[2]){
            dominated = true;
            break;
          }
          else if (P[i]->f[0] <= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] >= cu[2]){
            if (P[i]->f[0] > v[0])
              v[0] = P[i]->f[0];
          }
          else if (P[i]->f[0] >= cu[0] && P[i]->f[1] <= cu[1] && P[i]->f[2] >= cu[2]){
            if (P[i]->f[1] > v[1])
              v[1] = P[i]->f[1];
          }
          else if (P[i]->f[0] >= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] <= cu[2]){
            if (P[i]->f[2] > v[2])
              v[2] = P[i]->f[2];
          }
        }
        if (dominated)
          continue; //Cell's contribution is 0.
        Sminus = hvol3d(Pz, v, cl);
        double xslice = calculateslice(P, v, cl, 0);
        double yslice = calculateslice(Py, v, cl, 1);
        double zslice = calculateslice(Pz, v, cl, 2);
      //And then we integrate.
        for (int i=0;i<pdf.size();i++){
            double psi1 = exipsi(v[0],cl[0],pdf[i]->mu[0],pdf[i]->s[0]) - exipsi(v[0],cu[0],pdf[i]->mu[0],pdf[i]->s[0]);
            double psi2 = exipsi(v[1],cl[1],pdf[i]->mu[1],pdf[i]->s[1]) - exipsi(v[1],cu[1],pdf[i]->mu[1],pdf[i]->s[1]);
            double psi3 = exipsi(v[2],cl[2],pdf[i]->mu[2],pdf[i]->s[2]) - exipsi(v[2],cu[2],pdf[i]->mu[2],pdf[i]->s[2]);

            double gausscdf1 = gausscdf((cu[0]-pdf[i]->mu[0])/pdf[i]->s[0]) - gausscdf((cl[0]-pdf[i]->mu[0])/pdf[i]->s[0]);
            double gausscdf2 = gausscdf((cu[1]-pdf[i]->mu[1])/pdf[i]->s[1]) - gausscdf((cl[1]-pdf[i]->mu[1])/pdf[i]->s[1]);
            double gausscdf3 = gausscdf((cu[2]-pdf[i]->mu[2])/pdf[i]->s[2]) - gausscdf((cl[2]-pdf[i]->mu[2])/pdf[i]->s[2]);
            double sum = (psi1*psi2*psi3) - (Sminus*gausscdf1*gausscdf2*gausscdf3);
            //gausscdf represents chance of a point falling within the range [cl,cu)
            //psi = partial expected improvement
            //so psi - (gausscdf * (cl - v)) = p's expected distance from cl
            sum -= (xslice * gausscdf2 * gausscdf3 * (psi1 - (gausscdf1 * (cl[0]-v[0]))));
            sum -= (yslice * gausscdf1 * gausscdf3 * (psi2 - (gausscdf2 * (cl[1]-v[1]))));
            sum -= (zslice * gausscdf1 * gausscdf2 * (psi3 - (gausscdf3 * (cl[2]-v[2]))));
            if (sum > 0)
              answer[i] += sum;
        }
      }
    }
  }
  return answer;
}

vector<double> ehvi3d_sliceupdate(deque<shared_ptr<individual>> P, double r[DIMENSIONS], vector<shared_ptr<mus>> & pdf){
// vector<double> ehvi3d_sliceupdate(deque<individual*> P, double r[DIMENSIONS], vector<mus*> & pdf){
			
double mu1;
double mu2;
double mu3;
double s1;
double s2;
double s3;
double psi1;
double psi2;
double psi3;
double ex1;
double ex2;
double ex3;
double sum;
			
//Slice-update scheme.
  vector<double> answer; //The eventual answer.
  // specialind *newind;
  int n = P.size(); //Holds amount of points.
  thingy *Pstruct; //2D array with information about the shape of the dominated hypervolume
  deque<shared_ptr<specialind>> Px, Py, Pz; //P sorted by x/y/z coordinate with extra information.
  // deque<specialind*> Px, Py, Pz; //P sorted by x/y/z coordinate with extra information.
  double cellength[DIMENSIONS] = {0};
  while (answer.size() < pdf.size())
    answer.push_back(0);
  try{
    //Create sorted arrays which also contain extra information allowing the location in
    //the other sorting orders to be ascertained in O(1).
    sort(P.begin(), P.end(), xcomparator);
    for (unsigned int i=0;i<n;i++){
      // newind = new specialind;
      auto newind = make_shared<specialind>();
      newind->point = P[i];
      newind->xorder = i;
      Px.push_back(newind);
      Py.push_back(newind);
      Pz.push_back(newind);
    }
    sort(Py.begin(), Py.end(), specialycomparator);
    for (unsigned int i=0;i<n;i++){
      Py[i]->yorder = i;
    }
    sort(Pz.begin(), Pz.end(), specialzcomparator);
    for (unsigned int i=0;i<n;i++){
      Pz[i]->zorder = i;
    }
    //Then also reserve memory for the structure array.
    Pstruct = new thingy[n*n];
    for (int k=0;k<n*n;k++){
      Pstruct[k].slice = 0;
      Pstruct[k].chunk = 0;
      Pstruct[k].highestdominator = -1;
      Pstruct[k].xlim = 0;
      Pstruct[k].ylim = 0;
    }
  }
  catch (...) {
    cout << "An exception was thrown. There probably isn't enough memory available." << endl;
    cout << "zero-vector will be returned." << endl;
    return answer;
  }
  //Now we establish dominance in the 2-dimensional slices. Note: it is assumed that
  //P is mutually nondominated. This implementation of that step is O(n^3).
  for (int i=0;i<n;i++){
    for (int j=Pz[i]->yorder;j>=0;j--)
      for (int k=Pz[i]->xorder;k>=0;k--){
        Pstruct[k+j*n].highestdominator = i;
      }
    for (int j=Px[i]->zorder;j>=0;j--)
      for (int k=Px[i]->yorder;k>=0;k--){
        Pstruct[k+j*n].xlim = Px[i]->point->f[0] - r[0];
      }
    for (int j=Py[i]->zorder;j>=0;j--)
      for (int k=Py[i]->xorder;k>=0;k--){
        Pstruct[k+j*n].ylim = Py[i]->point->f[1] - r[1];
      }
  }
 
		
	/* cout << "I am here !!!" << endl; */
	int ii; int jj;
	for(ii=0;ii<=n;ii++)
	{
		for(jj=0;jj<=pdf.size();jj++)
		{
		flag1[ii][jj]=1;
		flag2[ii][jj]=1;
		flag3[ii][jj]=1;
		fpsi1[ii][jj]=1;
		fpsi2[ii][jj]=1;
		fpsi3[ii][jj]=1;
		}
	}
  //And now for the actual EHVI calculations.
  for (int z=0;z<=n;z++){
    //Recalculate Pstruct for the next 2D slice.
    if (z>0)
      for (int i=0;i<n*n;i++){
        Pstruct[i].chunk += Pstruct[i].slice * cellength[2];
      }
    //This step is O(n^2).
    for (int y=0;y<n;y++){
      for (int x=0;x<n;x++){
        if (Pstruct[x+y*n].highestdominator < z){ //cell is not dominated

          if (x > 0 && y > 0){
            Pstruct[x+y*n].slice = (Pstruct[x+(y-1)*n].slice - Pstruct[(x-1)+(y-1)*n].slice) + Pstruct[(x-1)+y*n].slice;
          }
          else if (y > 0){
            Pstruct[x+y*n].slice = Pstruct[x+(y-1)*n].slice;
          }
          else if (x > 0){
            Pstruct[x+y*n].slice = Pstruct[(x-1)+y*n].slice;
          }
          else
            Pstruct[x+y*n].slice = 0;
        }
        else {
          Pstruct[x+y*n].slice = (Px[x]->point->f[0] - r[0]) * (Py[y]->point->f[1] - r[1]);
        }
      }
    }
    //Okay, *now* we are going to calculate the EHVI, for real.

    for (int y=0;y<=n;y++){
      for (int x=0;x<=n;x++){
        /* double cl[DIMENSIONS], cu[DIMENSIONS]; //Boundaries of grid cells */
        double cl0 = (x == 0 ? r[0] : Px[x-1]->point->f[0]);
        double cl1 = (y == 0 ? r[1] : Py[y-1]->point->f[1]);
        double cl2 = (z == 0 ? r[2] : Pz[z-1]->point->f[2]);
        double cu0 = (x == n ? INFINITY : Px[x]->point->f[0]);
        double cu1 = (y == n ? INFINITY : Py[y]->point->f[1]);
        double cu2 = (z == n ? INFINITY : Pz[z]->point->f[2]);
        cellength[0] = cu0 - cl0;
        cellength[1] = cu1 - cl1;
        cellength[2] = cu2 - cl2;
        if (cellength[0] == 0 || cellength[1] == 0 || cellength[2] == 0 || (x < n && y < n && Pstruct[x+y*n].highestdominator >= z))
          continue; //Cell is dominated or of size 0.
        //We have easy access to Sminus and zslice because they are part of Pstruct.
        //xslice and yslice can be calculated from Pstruct->chunk.
        double slice[DIMENSIONS], Sminus, v[DIMENSIONS];
        if (x > 0 && y > 0){
          Sminus = Pstruct[(x-1)+(y-1)*n].chunk;
          slice[0] = (x == n ? 0 : (Pstruct[x+(y-1)*n].chunk - Sminus) / cellength[0]);
          slice[1] = (y == n ? 0 : (Pstruct[(x-1)+y*n].chunk - Sminus) / cellength[1]);
          slice[2] = Pstruct[(x-1)+(y-1)*n].slice;
        }
        else {
          Sminus = 0;
          slice[0] = ((y == 0 || x == n) ? 0 : (Pstruct[x+(y-1)*n].chunk - Sminus) / cellength[0]);
          slice[1] = ((x == 0 || y == n) ? 0 : (Pstruct[(x-1)+y*n].chunk - Sminus) / cellength[1]);
          slice[2] = 0;
        }

        if (y == n || z == n)
          v[0] = 0;
        else
          v[0] = Pstruct[y+z*n].xlim;
        if (x == n || z == n)
          v[1] = 0;
        else
          v[1] = Pstruct[x+z*n].ylim;
        if (x == n || y == n)
          v[2] = 0;
        else
          v[2] = (Pstruct[x+y*n].highestdominator == -1 ? 0 : (Pz[Pstruct[x+y*n].highestdominator]->point->f[2] - r[2]));
		 int psize=pdf.size();
		 double psiprod;
        for (int i=0;i<psize;i++){
			 mu1=pdf[i]->mu[0];
			 mu2=pdf[i]->mu[1];
			 mu3=pdf[i]->mu[2];
			 s1=pdf[i]->s[0];
			 s2=pdf[i]->s[1];
			 s3=pdf[i]->s[2];
			 
			 if (fpsi1[x][i])
               {exipsi1[x][i] = exipsi(r[0],cl0,mu1,s1) - exipsi(r[0],cu0,mu1,s1);fpsi1[x][i]=0;}
			 if (exipsi1[x][i] <0.000000000000001){break;}
             else 
			 { if (fpsi2[y][i])
               {exipsi2[y][i] = exipsi(r[1],cl1,mu2,s2) - exipsi(r[1],cu1,mu2,s2);fpsi2[y][i]=0;}
				psi2 = exipsi(r[1],cl1,mu2,s2) - exipsi(r[1],cu1,mu2,s2);
				if (exipsi2[y][i]<0.000000000000001){break;}
				else 
				{
				 if (fpsi3[z][i])
                    {exipsi3[z][i] = exipsi(r[2],cl2,mu3,s3) - exipsi(r[2],cu2,mu3,s3);fpsi3[z][i]=0;}
					psi3 = exipsi(r[2],cl2,mu3,s3) - exipsi(r[2],cu2,mu3,s3);
					if (exipsi3[z][i]<0.000000000000001){break;}
					else 
					{
							psiprod=exipsi1[x][i]*exipsi2[y][i]*exipsi3[z][i];
							 if (psiprod>0.000000000000001)
							 {
							 if (flag1[x][i])
								{gausscdf1[x][i] = gausscdf((cu0-mu1)/s1) - gausscdf((cl0-mu1)/s1);flag1[x][i] = 0;}
							 if (flag2[y][i])
								{gausscdf2[y][i] = gausscdf((cu1-mu2)/s2) - gausscdf((cl1-mu2)/s2);flag2[y][i] = 0;}
							 if (flag3[z][i])
								{gausscdf3[z][i] = gausscdf((cu2-mu3)/s3) - gausscdf((cl2-mu3)/s3);flag3[z][i] = 0;}

							 ex1 = exipsi1[x][i] - (gausscdf1[x][i] * (cl0-r[0]));
							 ex2 = exipsi2[y][i] - (gausscdf2[y][i] * (cl1-r[1]));
							 ex3 = exipsi3[z][i] - (gausscdf3[z][i] * (cl2-r[2]));

							sum = psiprod - (Sminus*gausscdf1[x][i]*gausscdf2[y][i]*gausscdf3[z][i])
							- (slice[0] * gausscdf2[y][i] * gausscdf3[z][i] * ex1)-v[0] * ex2 * ex3 * gausscdf1[x][i]
							- (slice[1] * gausscdf1[x][i] * gausscdf3[z][i] * ex2)-v[1] * ex1 * ex3 * gausscdf2[y][i]
							- (slice[2] * gausscdf1[x][i] * gausscdf2[y][i] * ex3)-v[2] * ex1 * ex2 * gausscdf3[z][i];
							
							
								if (sum >0){answer[i] += sum;}
							}
						}
					}	
				}
			
        }
      }
    }
  }
  delete [] Pstruct;
  return answer;
}

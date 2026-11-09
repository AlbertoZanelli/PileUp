/** 
 * @file FakePulse.C
 * @brief Create a mock pulse and fit the same pulse 
 * @author Roberto Serino
 * @date 10/04/2024
**/
 

#include <math.h>
#include "TApplication.h"
// #include <gsl/gsl_errno.h>
// #include <gsl/gsl_matrix.h>
// #include <gsl/gsl_odeiv.h>
#include "TMath.h"
#include <algorithm> // For std::max_element and std::min_element
#include "TComplex.h"
#include "TF1.h"
#include <unistd.h> 
#include "TCanvas.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include <iostream>
#include <algorithm>
#include "TFitResultPtr.h"
#include "TFitResult.h"
#include "TMatrixDSym.h"
#include "TVectorD.h"
#include "Math/MinimizerOptions.h"
#include <TRandom3.h>
#include <TTimeStamp.h>
#include <tuple>
#include <vector>
#include <iomanip>
#include <map>
#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>
#include <TText.h>
#include <fstream>
#include <sstream>


using namespace std;


double FitFakePulse(double *x, double *par) //0->t0, 1-> baseline, 2-> tilt, 3-> preamp, 4-> amp, 5...5+nzer-1-> zeri, 5+nzer...5+nzer+npol-1->poli (gli ultimi due sono sigma e omega del polo CC)
{

    double * new_par = par;
    int npol = static_cast<int>(new_par[0]);
    int nzer = static_cast<int>(new_par[1]);



    par = nullptr;
    double provvisorio[npol+nzer+5];
    
    for( int i = 2; i<npol+nzer+5+2; i++){
        provvisorio[i-2] = new_par[i]; 
    }
    par = provvisorio;



    //double value = par[2+npol+nzer];
                                        //  *par[2+nzer]
    double value = par[4+npol+nzer]*(exp(x[0]-par[0]))+par[2+npol+nzer]+par[3+npol+nzer]*(x[0]-par[0]);
    
    
	
    double Res[npol];

    if((x[0]>par[0]))
	{   
        
		for (int i=0;i<npol-2;i++) // Residui nel polo (per i poli reali)
		{
			Res[i]=1.;
			for (int j=0;j<nzer;j++)
				Res[i]*=(par[2+nzer+i]-par[2+j]); //numeratore(p1-z1)(p1-z2)...
			for (int j=0;j<npol-2;j++)
				if (i!=j) Res[i]/=(par[2+nzer+i]-par[2+nzer+j]); //denominatore (p1-p2)(p1-p3)...
			Res[i]/=(pow((par[2+nzer+i]-par[0+nzer+npol]),2)+par[1+nzer+npol]*par[1+nzer+npol]);
            	
		}
       

		Res[npol-2]=1.; 				//Residuo nel polo (per i due poli complessi coniugati)
		for (int j=0;j<nzer;j++)
			Res[npol-2]*=(pow((par[0+nzer+npol]-par[2+j]),2)+par[1+nzer+npol]*par[1+nzer+npol]);
            
       
		for (int j=0;j<npol-2;j++)
			Res[npol-2]/=(pow((par[0+nzer+npol]-par[2+nzer+j]),2)+par[1+nzer+npol]*par[1+nzer+npol]);
		Res[npol-2]/=(par[1+nzer+npol]*par[1+nzer+npol]);
		Res[npol-2]=TMath::Sqrt(Res[npol-2]);

    


		TComplex phi= - TComplex::I(); //fase
		for (int j=0;j<nzer;j++)
			phi*=(par[0+nzer+npol] + TComplex::I()*par[1+nzer+npol] - par[2+j]);
		for (int j=0;j<npol-2;j++)
			phi/=(par[0+nzer+npol] + TComplex::I()*par[1+nzer+npol] - par[2+nzer+j]);
		phi/=(2.0*par[1+nzer+npol]);

        

		double value1=0.;
		for (int i=0;i<npol-2;i++)
			value1+=Res[i]*exp((x[0]-par[0])*par[2+nzer+i]);
        value1+=Res[npol-2]*exp((x[0]-par[0])*par[0+nzer+npol])*cos(par[1+nzer+npol]*(x[0]-par[0]) + atan2(phi.Im(),phi.Re()));

        
		value +=  200*par[1]* value1;

        
		
        
    }
    
    return value;
}

double fitfuncNpMz(double *x, double *par) //0->t0, , 1-> amp, 5...5+nzer-1-> zeri, 5+nzer...5+nzer+npol-1->poli  -> baseline, -> tilt, -> preamp
{
    
    double * new_par = par;
    int npol = static_cast<int>(new_par[0]);
    int nzer = static_cast<int>(new_par[1]);


    par = nullptr;
    double provvisorio[npol+nzer+5];
    
    for( int i = 2; i<npol+nzer+5+2; i++){
        provvisorio[i-2] = new_par[i]; 
    }
    par = provvisorio;



    //double value = par[2+npol+nzer];

    double value = par[4+npol+nzer]*(exp(x[0]*par[2+nzer]))+par[2+npol+nzer]+par[3+npol+nzer]*(x[0]-par[0]);
    
    double Res[npol];
		
	
	if(x[0] > par[0])
	{
		for (int i=0;i<npol;i++) // Residui nel polo
		{
			Res[i]=1;
			for (int j=0;j<nzer;j++)
				Res[i]*=(par[2+nzer+i]-par[2+j]);
			for (int j=0;j<npol;j++)
				if (i!=j) Res[i]/=(par[2+nzer+i]-par[2+nzer+j]);

		}
		
		double value1=0.;
		for (int i=0;i<npol;i++)
			value1+=Res[i]*TMath::Exp((x[0]-par[0])*par[2+nzer+i]);
        
        value1*=/*pow(-1,(npol+nzer-1))*/par[1];
		
        value+=value1;
			
	}

    return value;
}

void ReadVectorsFromFile(const std::string& filename, std::vector<double>& x, std::vector<double>& y) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << "!" << std::endl;
        return;
    }

    std::string line;
    bool is_fx1 = false, is_fy1 = false;

    while (std::getline(file, line)) {
        // Skip any empty lines or comments
        if (line.empty() || line[0] == '/') {
            continue;
        }

        // Check which array we are reading
        if (line.find("Double_t Graph0_fx") != std::string::npos) {
            is_fx1 = true;
            is_fy1 = false;
        } else if (line.find("Double_t Graph0_fy") != std::string::npos) {
            is_fy1 = true;
            is_fx1 = false;
        }

        // Process data for Graph0_fx1
        if (is_fx1 && line.find('{') != std::string::npos) {
            while (std::getline(file, line)) {
                if (line.find("}") != std::string::npos) {
                    break; // Stop when we reach the closing brace
                }
                std::istringstream iss(line);
                std::string token;
                while (iss >> token) {
                    // Clean up token (remove commas, spaces, etc.)
                    token.erase(std::remove(token.begin(), token.end(), ','), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '{'), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '}'), token.end());

                    try {
                        x.push_back(std::stod(token)); // Add to x vector
                    } catch (...) {
                        // Ignore invalid tokens
                    }
                }
            }
        }

        // Process data for Graph0_fy1
        if (is_fy1 && line.find('{') != std::string::npos) {
            while (std::getline(file, line)) {
                if (line.find("}") != std::string::npos) {
                    break; // Stop when we reach the closing brace
                }
                std::istringstream iss(line);
                std::string token;
                while (iss >> token) {
                    // Clean up token (remove commas, spaces, etc.)
                    token.erase(std::remove(token.begin(), token.end(), ','), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '{'), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '}'), token.end());

                    try {
                        y.push_back(std::stod(token)); // Add to y vector
                    } catch (...) {
                        // Ignore invalid tokens
                    }
                }
            }
        }
    }

    file.close();
}

double computeRMS(const std::vector<double>& y, int index_fin) {
    if (index_fin <= 0) return 0.0;

    // Compute the mean of the values up to index_fin
    double sum = 0.0;
    for (int i = 0; i < index_fin; ++i) {
        sum += y[i];
    }
    double mean = sum / index_fin;

    // Compute the RMS (Root Mean Square)
    double sum_of_squares = 0.0;
    for (int i = 0; i < index_fin; ++i) {
        sum_of_squares += (y[i] - mean) * (y[i] - mean);
    }

    return std::sqrt(sum_of_squares / index_fin);
}


int main(int argc, char *argv[]) {

    // Convert command-line arguments to integers and assign them to global variables
    int npol = std::atoi(argv[1]);
    int nzer = std::atoi(argv[2]);
    int cc = std::atoi(argv[3]);
    std::string filename = argv[4];

    TApplication *app = nullptr;
    if (!gApplication) {
        app = new TApplication("app", &argc, argv);
    } else {
        app = gApplication;
    }

    double pretrigger = 0.3; // Example pretrigger value


    
    
    std::vector<double> x, y;

    // Read data from the file
    ReadVectorsFromFile(filename, x, y);

    


    double t_0 = 0.28;
    double t_fin = 0.39;
    double min_diff = std::numeric_limits<double>::max();
    int closest_index = -1;



    // Find the closest value to 0.28
    for (size_t i = 0; i < x.size(); ++i) {
        double diff = std::abs(x[i] - t_0);
        if (diff < min_diff) {
            min_diff = diff;
            closest_index = i;
        }
    }

    int index_0 = closest_index;

    min_diff = std::numeric_limits<double>::max();
    closest_index = -1;

    // Find the closest value to 0.28
    for (size_t i = 0; i < x.size(); ++i) {
        double diff = std::abs(x[i] - t_fin);
        if (diff < min_diff) {
            min_diff = diff;
            closest_index = i;
        }
    }

    int index_fin = closest_index;

     // Create and plot the TGraph
    TGraphErrors* graph = new TGraphErrors(x.size(), x.data(), y.data());


    graph->SetTitle("graph;Time(s);Signal (V)");

   
    double rms = computeRMS(y, index_fin);
    cout<<"RMS: "<<rms<<endl;

    for(int i=0;i<x.size();i++)
        graph->SetPointError(i,0,rms);
    
    double max = *std::max_element(y.begin(), y.end());
    double min = *std::min_element(y.begin(), y.end());

    double MaxMin = max-min;

    if(npol==5)
        MaxMin=MaxMin*10000;
    

    // Draw the histogram on the canvas
    TCanvas *c1 = new TCanvas("c1", "Subset Histogram", 800, 600);
    TGraph *graphWithoutErrors = new TGraph(x.size(), x.data(), y.data());  // Create a TGraph without errors
    graphWithoutErrors->SetTitle(";Time(s);Signal (V)");

    graphWithoutErrors->GetXaxis()->SetLimits(t_0, t_fin);
    graphWithoutErrors->Draw("AL");  // "A" for axis, "P" for points
    graphWithoutErrors->SetLineWidth(3);     // Set line width



    TF1*     fBolometerModel;

    if(cc==1)
    {
        fBolometerModel = new TF1("BolometerModel",&FitFakePulse, 0, 0.7, 12);
    }
    else
    {
        fBolometerModel = new TF1("BolometerModel",&fitfuncNpMz, 0, 0.7, 12);
    }


    fBolometerModel->SetParName(0,"Npol");
    fBolometerModel->FixParameter(0,npol);


    fBolometerModel->SetParName(1,"Nzer");
    fBolometerModel->FixParameter(1,nzer);

    fBolometerModel->SetParName(2,"t_0");
    fBolometerModel->SetParameter(2,pretrigger-0.01);
    fBolometerModel->SetParLimits(2,0.0,0.5);

    
    fBolometerModel->SetParName(3,"Amplitude");
    fBolometerModel->SetParameter(3,MaxMin*3500);
    fBolometerModel->SetParLimits(3,1,1000000);
    //fBolometerModel->FixParameter(3,7.51912e+04 );	

    
    for (int j=0; j<nzer; j++)
    {
        fBolometerModel->SetParameter(4+j,-117.141*j);						//Zeri
        fBolometerModel->SetParLimits(4+j,-10000.,-0.001);
        //fBolometerModel->FixParameter(4+j,-2.14436e+00);

    }



    for (int j=0; j<npol; j++)
    {
        if(j==0)
        {
            fBolometerModel->SetParameter(4+nzer+j,-62.1311);				//Poli
            fBolometerModel->SetParLimits(4+nzer+j,-10000,-0.001);
        }    
        else
        {
            fBolometerModel->SetParameter(4+nzer+j,-260*j*0.5);				//Poli
            fBolometerModel->SetParLimits(4+nzer+j,-10000.0,-0.001);
        }
    }

    if (cc==1)
    {					
        fBolometerModel->SetParameter(3+nzer+npol,600.0);				
        fBolometerModel->SetParLimits(3+nzer+npol,0.00001,10000.0);
    }



    fBolometerModel->SetParName(4+npol+nzer,"Baseline");
    fBolometerModel->SetParameter(4+npol+nzer,min);                                 //Baseline
    fBolometerModel->SetParLimits(3,min*0.000001,min*10000000);

    fBolometerModel->SetParName(5+npol+nzer,"Tilt");
    fBolometerModel->SetParameter(5+npol+nzer,0.0);
    fBolometerModel->SetParLimits(5+npol+nzer,-10000,10000);								//Tilt

    fBolometerModel->SetParName(6+npol+nzer,"PreAmp");
    // fBolometerModel->SetParameter(5, 0.0);								//preAmp
    // fBolometerModel->SetParLimits(5,0,5.0);
    fBolometerModel->FixParameter(6+npol+nzer,0.0);


    fBolometerModel->Draw("same");
    fBolometerModel->SetLineWidth(3);
    fBolometerModel->SetLineColor(kRed);

    fBolometerModel->SetNpx(800); // Set to 1000 points (or higher, as needed)




    TFitResultPtr fitResult = graph->Fit(fBolometerModel, "RSQ", "", t_0, t_fin);
    
    // Update the canvas
    c1->Update();

    fitResult->Print();


    

    TGraphErrors *residualsGraph = new TGraphErrors();
    residualsGraph->SetTitle(";Time(s);Signal (V)");
    int pointIndex = 0; // Initialize a point index
    for (int i = index_0; i <= index_fin; ++i) {
        double modelValue = fBolometerModel->Eval(x[i]);
        double residual = y[i] - modelValue;
        residualsGraph->SetPoint(pointIndex++, x[i], residual); // Use the pointIndex variable
        //residualsGraph->SetPointError(pointIndex++, 0.001, RMS*10); // Set constant y-error as RMS

    }


    residualsGraph->GetXaxis()->SetLimits(t_0, t_fin); 
    

    // Create a new canvas
    TCanvas *c2 = new TCanvas("c2", "Residuals", 800, 600);

    // Divide the canvas into two pads
    c2->Divide(1, 2);

    // Upper pad for the original graph
    c2->cd(1);
    graphWithoutErrors->Draw("AL");
    graphWithoutErrors->SetLineWidth(3);
    fBolometerModel->Draw("same");
    fBolometerModel->SetLineWidth(3);
    fBolometerModel->SetLineColor(kRed);
    fBolometerModel->SetNpx(800);

    // Display the RMS on the canvas
    TText* text = new TText(0.5, 0.9, Form("RMS: %.4f", rms));
    text->SetNDC(); // Use normalized coordinates (0 to 1)
    text->SetTextSize(0.04);
    text->SetTextColor(kBlack);
    text->Draw();

    // Lower pad for the residuals graph
    c2->cd(2);
    residualsGraph->SetLineColor(kRed); // Set the line color to red (or any other color you prefer)
    residualsGraph->SetLineWidth(3); // Set the line width
    residualsGraph->Draw("AL"); // Draw the residuals as a continuous line




    // Draw the canvas
    c2->Draw();

    //Save the canvas
    //c2->SaveAs("NTD_with_residuals.root");

    app->Run();

    return 0;
}
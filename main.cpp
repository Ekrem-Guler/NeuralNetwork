#include <iostream>
#include <vector>
#include <string>
#include <cstdlib> 
#include <cmath>
#include <numeric>
#include <functional>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <algorithm>
#define N 4





using namespace std;
int input_shape[2]= {5,6000};
int sizes[N] = {input_shape[0],8,5,1};

class Layer{
	public:
		// activation'larýn tutulduðu vektör ve weightslerin tutulduðu vektör oluþturuluyor.
		vector<double>activation_values;	
		vector<vector <double> > weights;
		vector<double> bias;
		vector<double> w;
		// weightslerimizin hepsine deðerini atýyoruz.
		vector<vector <double> > create_weights(int s){
			
			vector<vector <double> > weights;
			// Hangi layerda isek ona giden weightsleri oluþturduðumuz için gelen layerýn nöron sayýsý*gidilen layerýn nöron sayýsý kadar weight olmasý gerekir.
			// Her gittiðimiz nöron için bir vector oluþturuldu. Yani [[j tane weights]...[j tane weights]] i tane olmak üzere her nöronun kendisine gelen weightsleri var.
			for(int i= 1;i<=sizes[s];i++){	
				w.clear();
				bias.push_back(0);
				for(int j= 0;j<sizes[s-1];j++){
					w.push_back(1);	
				}
					weights.push_back(w);
			}
		return weights;
		}
		
		Layer(int s){
			// Sonuncu için weights olmayacak 3 tane weights olacak her layerdan giden weightsleri temsil ediyor.
			if(s!=N){
				weights = create_weights(s);
			}
		}
		
};

class NeuralNetwork{
	public:
		//Activation fonksiyonumuzun ne olacaðý seçiliyor.
		double activation(string s,double z){
			if(s == "linear"){
				return z;
			}
			else return 1/(1+exp(-z));
		}
		// Türev alýrken gerekeceði için sigmoid'in türevini alýyoruz.
		double sigmoid_derivative(double a){
			return a*(1-a);
		}
		// forward propagation yaparak tahminimizi alýyoruz.
		vector<Layer> forward_pass(vector<Layer> Layers_vec,int num){
			//Input layerýndan baþlayarak output layerýn bir þey ile çarpýlmasý gerekmediðinden N-1 kez döndürüyoruz.
			//dot product yapacaðýmýz için geldiðimiz nörp-onlar ile gidilen nöronun weightslerini inner_product fonksiyonu kullanarak çarpýyoruz.
			for(int i= 0;i<N-1;i++){
				for(int j= 0;j<sizes[i+1];j++){
								
					double dot = inner_product(Layers_vec[i].weights[j].begin(),Layers_vec[i].weights[j].end(),Layers_vec[i].activation_values.begin(), 0.0);	 
					// Ýlk kez çalýþtýrdýðýmýzda sonraki nöronlara atama yapacak, birden çok kez döndürdüðümüzde nöronlarýn deðerini deðiþtirecek.
					if(num==0){
						
						Layers_vec[i+1].activation_values.push_back(activation("sigmoid",dot+Layers_vec[i].bias[j]));
					}
					else {
						if(i!=N-2){
							Layers_vec[i+1].activation_values[j] = activation("sigmoid",dot+Layers_vec[i].bias[j]);
						}
						else Layers_vec[i+1].activation_values[j] = activation("sigmoid",dot+Layers_vec[i].bias[j]);
					}
				}
			}
			return Layers_vec;
		}
		
		// cost fonksiyonumuzun türevi
		double cost_derivative(double reel_y,double pred_y){
		 	return pred_y-reel_y;
		}
		
		// weightsleri deðiþtirmek için cost fonksiyonumuzun o weightse göre türevini almamýz lazým. Zincir kuralý uygulayarak bunu yapabiliriz.
		// Layers_vec ana layer vectorümüz ve change_layer olmasý gereken weightslerin toplandýðý vektör olmak üzere iki tane parametremiz bulunuyor.
		vector<Layer> sum_of_derivatives(vector<Layer> Layers_vec, vector<Layer> change_layer){
		 // Weightsler N-1 parçadan oluþuyor ve output layerlar diðer layerlardan ayrý çalýþýr.0'ý da dahil edersek N-1 kez döner.
			for(int i= N-2;i>=0;i--){
				//Kaç tane weight var ise o kadar dönmeli.
				for(int j=0;j<Layers_vec[i].weights.size();j++){
					
					for(int k= 0;k<Layers_vec[i].weights[j].size();k++){
						// Son layerda ise change_layerýmýz kendisi, gittiði nöron ve sigmoid türevinin çarpýmý olur.
						if(i==N-2){
							
							change_layer[i].weights[j][k] =1;
						}
						// Eðer son layerda deðilse 2 sonraki nöronlarýn deðerleri output layera kadar olan çarpýmlarý verecektir bize.
						// Eðer onlarý çarpýldýklarý aktivasyon nöronuna ve sigmoid türevlerine bölersek ardýndan da gereken aktivasyon nöronu sig_der ile çarparsak deðeri buluruz.
						else{
							
							for(int m=0;m<Layers_vec[i+2].activation_values.size();m++){
							
								double mul = change_layer[i+1].weights[m][k]*Layers_vec[i+1].weights[m][k]; 
								
								change_layer[i].weights[j][k] += mul;
								
							}
						}
					}
				}
			}
			return change_layer;
		}
		vector<Layer> backpropagation(vector<Layer> Layers_vec,vector<Layer> change_layer,double learning_rate,double cost_derivative,double sigmoid_derivative,string s){
			for(int i=0;i<Layers_vec.size()-1;i++){
				
				for(int j=0;j<Layers_vec[i].weights.size();j++){
					
					for(int k= 0;k<Layers_vec[i].weights[j].size();k++){
						if(s == "sigmoid"){
							Layers_vec[i].weights[j][k] -= learning_rate*cost_derivative*change_layer[i].weights[j][k]*sigmoid_derivative*Layers_vec[i].activation_values[k];
							
						}
						else if(s == "linear") Layers_vec[i].weights[j][k] -= learning_rate*cost_derivative*change_layer[i].weights[j][k]*Layers_vec[i].activation_values[k];
					}
				}
			}
			return Layers_vec;
		}
		
		
		NeuralNetwork(int epoch,double learning_rate,vector<vector<double> > x,vector<double> y){
			vector<Layer> Layers_vec;
			vector<Layer> change_layer;
			for(int i= 1;i<=N;i++){
				Layers_vec.push_back(Layer(i));
				change_layer.push_back(Layer(i));
			}
			for(int j=0;j<input_shape[1];j++){
				
					for(int m = 0;m<x[j].size();m++){
				
						Layers_vec[0].activation_values.push_back(x[j][m]);
						
					}
				
				for(int i=0;i<epoch;i++){
					
					Layers_vec = forward_pass(Layers_vec,i);
					
					
					vector<Layer> change_layer1 = sum_of_derivatives(Layers_vec,change_layer);
					double cost = cost_derivative(y[j],Layers_vec[N-1].activation_values[0]);
					double sigmoid_derivativ = sigmoid_derivative(Layers_vec[N-1].activation_values[0]);
					
					Layers_vec = backpropagation(Layers_vec,change_layer1,learning_rate,cost,sigmoid_derivativ,"sigmoid");
					cout<<i<<"    "<<Layers_vec[N-1].activation_values[0]<<"  "<<cost<<"  "<<sigmoid_derivativ<<endl;

					if(cost == 0) break;
				}
				cout<<0<<"    "<<Layers_vec[N-1].activation_values[0]<<endl;
				for(int i= 0;i<N;i++){
					Layers_vec[i].activation_values.clear();
				}
				
		    }	
		}
};	

void single_line_dummy(vector<vector<double> > x,vector<double> y){
	
	vector<double> a;
	a.push_back(0.3);
	a.push_back(0.4);
	a.push_back(0.5);
	y.push_back(0.9);
	x.push_back(a);
	NeuralNetwork(1000,10,x,y);
}
int main(){
	vector<vector<double> > x;
	vector<double> y;
	single_line_dummy(x,y);
	// NeuralNetwork(1001,0.0000001);
}




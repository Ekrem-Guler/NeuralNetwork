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
#include <random>
#define N 4
using namespace std;

struct Value {
    double data;
    double grad = 0.0;
};

struct Neuron {
    double value;
    double activation_value;
    double grad = 0.0;

    Neuron(double v = 0) : value(v), activation_value(v) {}
};

class Input{
    public:
        vector<vector<double> > v;
        Input(vector<vector<double> > vec){
            v = vec;
        }
};

class Layer{
    public:
        
        vector<vector<Value>> weights;
        vector<Value> biases;
        vector<Neuron> neurons;
        int unit11;
        int previous,current;
        bool works_before = false;
        vector<vector<Value>> first_weights(){
            static mt19937 gen(42);
            normal_distribution<double> dist(0.0, 0.5);
            for(int i=0;i<current;i++){
                vector<Value> row;
                for(int j= 0;j<previous;j++){
                    Value v;
                    v.data =dist(gen);
                    v.grad = 0.0;
                    row.push_back(v);
                    
                }
                weights.push_back(row);
            }
            return weights;
        }
        
        Layer(int inc,int outc){
            previous = inc;
            current = outc;
            neurons.resize(current);
            if(works_before == false){
                first_weights();
            }
        }
};

class Sequence{
    public:
        double activation_func(double value,string activation){
            if(activation == "sigmoid"){
                return 1/(1+exp(-value));
            }
            if(activation == "ReLU"){
                return max(0.0,value);
            }
            return value;
        }
        double activation_der(double value,string activation){
            if(activation == "sigmoid"){
                return (1-value)*value;
            }
            if(activation == "ReLU"){
                
                return value > 0 ? 1.0 : 0.0;
            }
            return 1.0;
        }
        void forward(int current_s,int previous_s,Layer &previous_l,Layer &current_l,string activation,bool works_before){
                double result;
                for(int i= 0;i<current_s;i++){
                    
                    double sum = 0.0;
                    for(int j=0;j<previous_s;j++){
                        
                        sum+=previous_l.neurons[j].activation_value*current_l.weights[i][j].data;
                        
                    }
                    result = activation_func(sum,activation);

                    current_l.neurons[i].activation_value = result;

                    
                }
        }
            void backward(int current_s,int previous_s,Layer &previous_l,Layer &current_l,string activation){
                vector<double>delta(current_s);
                for(int j=0;j<current_s;j++){
                    delta[j] = current_l.neurons[j].grad*activation_der(current_l.neurons[j].activation_value,activation);
    /*cout<<"DELTAS: "<<delta[j]<<endl;*/
                }

                for(int i=0;i<previous_s;i++){
                    double sum = 0.0;
                    
                    for(int j=0;j<current_s;j++){
                        
                        current_l.weights[j][i].grad += delta[j]*previous_l.neurons[i].activation_value;
                        sum += delta[j]*current_l.weights[j][i].data;
                    }
                    
                    
                    previous_l.neurons[i].grad = sum;
                    
                }
            }
        void optimizer(int current_s,int previous_s,Layer &current_l,double learning_rate,int batch_size){
            for(int i=0;i<current_s;i++){
                for(int j=0;j<previous_s;j++){
                    current_l.weights[i][j].data -= learning_rate*current_l.weights[i][j].grad/batch_size;

                }
                
            }
        }

        double loss_der(double pred,double y,string loss){
            cout<<loss;
            if(loss == "MSE"){
                cout<<2*(pred-y)<<endl;
                return 2*(pred-y);
            }
            return 1.0;
        }

        void one_step(vector<Layer> &l,vector<double> v,vector<double> y,string loss,int phase,string activate,double &loss_sum,vector<double> losses){
            int size_l = l.size();
            int size_inp = v.size();
            for (int i=0;i<size_inp;i++){
                l[0].neurons[i] = v[i];
                
            }
            
            l[0].works_before = true;

            for(int i= 1;i<size_l;i++){
                
                if(i==size_l-1){
                    
                    forward(l[i].current,l[i].previous,l[i-1],l[i],"simple",l[i].works_before);
                } 
                else {
                    forward(l[i].current,l[i].previous,l[i-1],l[i],activate,l[i].works_before);
                }
                l[i].works_before = true;
            }
            

cout<<"Answer for "<<phase<<":   "<<l[size_l-1].neurons[0].activation_value<<endl<<"Result:  " <<loss_sum<<endl<<"SEED: "<< l[size_l-1].neurons[0].grad <<endl;
            
                
            l[size_l-1].neurons[0].grad =loss_der(l[size_l-1].neurons[0].activation_value,y[phase],loss);

            for(int i = size_l-1;i>0;i--){
                string act = (i == size_l-1) ? "simple" : activate;
                backward(l[i].current,l[i].previous,l[i-1],l[i],act);
            }
        }

        void zero_grad(vector<Layer> &l){
            int size_l=l.size();
            for(int i=0;i<size_l;i++){
                for(int j=0;j<l[i].current;j++){
                    l[i].neurons[j].grad = 0.0;
                    
                    for(int k=0;k<l[i].previous;k++){
                        l[i].weights[j][k].grad = 0.0;
                    }
                }
            }   
        }

        void all_stages(vector<Layer> &l,Input inp,int batch_size,vector<double> y,string loss,int epoch,double lr,string activate){
            int size_l = l.size();
            int size_batch = inp.v.size()/batch_size;
            double loss_sum = 0.0;
            vector<double> losses = {0.0,0.0,0.0};
            for(int e=0;e<epoch;e++){
                cout<<e<<" Epoch: "<<endl;
                for(int i = 0;i<size_batch;i++){

                    
                        zero_grad(l);
                    
                    
                    for(int j= 0;j<batch_size;j++){
                        cout<<"INPUT: "<<inp.v[j+i*batch_size][0]<<endl;
                        
                        one_step(l,inp.v[j+i*batch_size],y,loss,j+i*batch_size,activate,loss_sum,losses);
                    }

                    for(int j = 1;j<size_l;j++){
                        optimizer(l[j].current,l[j].previous,l[j],lr,batch_size);
                    }
                }
            }
        }
        Sequence(vector<Layer> &l,Input inp,int batch_size,vector<double> y,string loss,int epoch,double lr,string activate){
            all_stages(l,inp,batch_size,y,loss,epoch,lr,activate);
          /*
 int size_l = l.size();
            int size_inp = inp.v[0].size();
            for (int i=0;i<size_inp;i++){

            
               
                l[0].neurons[i] = inp.v[0][i];
                
            }
// loss before
            for(int i= 1;i<l.size();i++){
                forward(l[i].current,l[i].previous,l[i-1],l[i],"simple",l[i].works_before);
                cout<<"ACTIVATION BEFORE: "<<l[i].neurons[0].activation_value<<endl;
            }
            l[size_l-1].neurons[0].grad = loss_der(l[size_l-1].neurons[0].activation_value,y[0],loss);
            for (int i = size_l-1; i > 0; i--)          // backward
                backward(l[i].current, l[i].previous, l[i-1], l[i], activate);
            double analytical = l[1].weights[0][0].grad;
            double saved = l[1].weights[0][0].data;
            double eps = 1e-4;
            cout<<"loss: "<< l[l.size()-1].neurons[0].activation_value - y[0];
            double loss_before = pow(l[l.size()-1].neurons[0].activation_value - y[0], 2);

            // perturbe et
            l[1].weights[0][0].data = saved + eps;
            for(int i= 1;i<l.size();i++){
                forward(l[i].current,l[i].previous,l[i-1],l[i],"simple",l[i].works_before);
                cout<<"ACTIVATION AFTER: "<<l[i].neurons[0].activation_value<<endl;
            }
            double loss_after = pow(l[size_l-1].neurons[0].activation_value - y[0], 2);

            l[1].weights[0][0].data = saved;  // geri al

            double numerical = (loss_after - loss_before) / eps;
            cout << "numerical: " << numerical << "  backprop: " << l[1].weights[0][0].grad << endl;
            */
        }
};


int main(){
    
    vector <vector<double>> v = {{1,2,3},{2,3,5},{3,4,7}};
    vector<double> y = {6, 10, 14}; 
    vector<vector<Neuron>> neurons1;
    vector<Neuron> neurons(v[0].begin(), v[0].end());
    neurons1.push_back(neurons);
    Input input = (v);
    int first_size = v[0].size();
    Layer layer1 = Layer(0,first_size);
    Layer layer = Layer(first_size,3);
    Layer layer2 = Layer(3,2);
    Layer layer3 = Layer(2,1);
    cout<<layer.weights[0].size();
    vector<Layer> layers = {layer1,layer,layer2,layer3};
    Sequence sequnce1 = Sequence(layers,input,1,y,"MSE",5000,0.001,"sigmoid");
    return 0;
}
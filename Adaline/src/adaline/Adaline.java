/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package adaline;

import java.util.*;
import java.io.*;
import static java.lang.Math.abs;
import java.text.*;
import java.math.*;
import java.util.Random;
/**
 *
 * @author bruno
 */
public class Adaline {

    /**
     * @param args the command line arguments
     */
    
    static double theta = 0;
    static int nTraining = 35;
    static double output;
    static int epoca = 0;
    static int saida[] = new int[15];
    static int amostra;
    
    static double learning_rate = 0.0025;
    static double weights[] = new double[5];
    static double globalErro;
    static double erro = 0.000001;
    static double EQM_atual = 1;
    static double EQM_ant = Double.POSITIVE_INFINITY;
    static double EQM = 0;
    static double EQMdiff = 0;
    static LinkedList error = new LinkedList();
    
    //Dados para treinamento
    static double x0[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static double x1[] = {0.4329, 0.3024, 0.1349, 0.3374, 1.1434, 1.3749, 0.7221, 0.4403, -0.5231, 0.3255, 0.5824, 0.1340, 0.1480, 0.7359, 0.7115, 0.8251, 0.1569, 0.0033, 0.4243, 1.0490, 1.4276, 0.5971, 0.8475, 1.3967, 0.0044, 0.2201, 0.6300, -0.2479, -0.3088, -0.5180, 0.6833, 0.4353, -0.1069, 0.4662, 0.8298};
    static double x2[] = {-1.3719, 0.2286, -0.6445, -1.7163, -0.0485, -0.5071, -0.7587, -0.8072, 0.3548, -2.0000, 1.3915, 0.6081, -0.2988, 0.1869, -1.1469, -1.2840, 0.3712, 0.6835, 0.8313, 0.1326, 0.5331, 1.4865, 2.1479, -0.4171, 1.5378, -0.5668, -1.2480, 0.8960, -0.0929, 1.4974, 0.8266, -1.4066, -3.2329, 0.6261, -1.4089};
    static double x3[] = {0.7022, 0.8630, 1.0530, 0.3670, 0.6637, 0.4464, 0.7681, 0.5154, 0.2538, 0.7112, -0.2291, 0.4450, 0.4778, -0.0872, 0.3394, 0.8452, 0.8825, 0.5389, 0.2634, 0.9138, -0.0145, 0.2904, 0.3179, 0.6443, 0.6099, 0.0515, 0.8591, 0.0547, 0.8659, 0.5453, 0.0829, 0.4207, 0.1856, 0.7304, 0.3119};
    static double x4[] = {-0.8535, 2.7909, 0.5687, -0.6283, 1.2606, 1.3009, -0.5592, -0.3129, 1.5776, -1.1209, 4.1735, 3.2230, 0.8649, 2.3584, 0.9573, 1.2382, 1.7633, 2.8249, 3.5855, 1.9792, 3.7286, 4.6069, 5.8235, 1.3927, 4.7755, 0.7829, 0.8093, 1.7381, 1.5483, 2.3993, 2.8864, -0.4879, -2.4572, 3.4370, 1.3235};
    static int y[] = {1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1};
    
    //Dados para teste
    static double data[][] = {{0.9694, 0.6909, 0.4334, 3.4965},{0.5427, 1.3832, 0.6390, 4.0352},{0.6081, -0.9196, 0.5925, 0.1016},{-0.1618, 0.4694, 0.2030, 3.0117},{0.1870, -0.2578, 0.6124, 1.7749},{0.4891, -0.5276, 0.4378, 0.6439},{0.3777, 2.0149, 0.7423, 3.3932},{1.1498, -0.4067, 0.2469, 1.5866},{0.9325, 1.0950, 1.0359, 3.3591},{0.5060, 1.3317, 0.9222, 3.7174},{0.0497, -2.0656, 0.6124, -0.6585},{0.4004, 3.5369, 0.9766, 5.3532},{-0.1874, 1.3343, 0.5374, 3.2189},{0.5060, 1.3317, 0.9222, 3.7174},{1.6375, -0.7911, 0.7537, 0.5515}};
    
    //Randomizar valores para os pesos
    public static double[] weightRandom(){
        Random rand = new Random();
        for(int count = 0; count < 5; count ++){
          weights[count] = rand.nextDouble();
        }        
        return weights;
    }
    
    //Calcular a saída Y simulada
    public static double outputCalc(double weights[], double x0, double x1, double x2, double x3, double x4){
        double sum =  x0*weights[0] + x1*weights[1] + x2*weights[2] + x3*weights[3] + x4*weights[4];
        /*
        int retorno;
        if(sum >= theta){
            retorno = 1;
        }else{
            retorno = -1;
        }
        */
        return sum;
    }
    
    //Calcular a saída Y pós treinamento
    public static int perceptronOutput(double theta, double weights[], double input1, double input2, double input3, double input4){
        double y =  -1*weights[0] + input1*weights[1] + input2*weights[2] + input3*weights[3] +input4*weights[4];        
        int retorno;
        if(y >= theta){
            retorno = 1;
        }else{
            retorno = -1;
        }
        return retorno;
        
    }
    
        
    public static void main(String[] args) {
        double pesos[] = weightRandom();        
        System.out.println("Pesos Iniciais: " + "\n" + "Peso w0 " + pesos[0] + "\n" + "Peso w1 "+ pesos[1] +"\n"+"Peso w2 " + pesos[2] + "\n" +"Peso w3 " + pesos[3] + "\n" + "Peso w4 " + pesos[4]);
        while(Math.abs(EQM_atual - EQM_ant) > erro){
            EQM_ant = EQM_atual;
         //Loop por tds entradas(1 época)
            for(int p = 0; p < nTraining; p++){
                
                //Obter saída calculada
                output = outputCalc(pesos, x0[p], x1[p], x2[p], x3[p], x4[p]);
                                
                //Atualização dos pesos
                pesos[0] = pesos[0] + learning_rate * (y[p]-output) *x0[p];
                pesos[1] = pesos[1] + learning_rate * (y[p]-output) *x1[p];
                pesos[2] = pesos[2] + learning_rate * (y[p]-output) *x2[p];
                pesos[3] = pesos[3] + learning_rate * (y[p]-output) *x3[p];
                pesos[4] = pesos[4] + learning_rate * (y[p]-output) *x4[p];                              
            
                EQM = EQM + ((y[p]-output)*(y[p]-output));
                //System.out.println("ESSA É A DIFERENÇA ENTRE SAIDAS "+ ((y[p]-output)*(y[p]-output)));
            }
         error.addLast(EQM_atual);
         
         EQM_atual = EQM/nTraining;
         epoca = epoca + 1;
         EQM = 0;
         EQMdiff = Math.abs(EQM_atual - EQM_ant);                 
        }
        
        System.out.println("Epocas:"+epoca+"\n"+ "Peso w0:" + weights[0] + "\n" + "Peso w1:" + weights[1] + "\n" + "Peso w2:" +weights[2]+ "\n"+"Peso w3:"+weights[3] + "\n" + "Peso w4:"+weights[4]);
        
        
        
        //Classificação das 15 amostras para 5 treinamentos
        for(int counter1 = 0; counter1 < 15; counter1++){
              
              saida[counter1] = perceptronOutput(theta, pesos, data[counter1][0], data[counter1][1], data[counter1][2], data[counter1][3]);
               
              amostra = counter1 + 1;
              
              System.out.println("Amostra: " +amostra+"saida:"+saida[counter1]+"\n");
        }
        for(int i = 0; i < error.size(); i++ ){
        System.out.println(error.get(i)+ " ");
        }
    }
    
}

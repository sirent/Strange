/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package iseng;

import java.util.Scanner;

/**
 *
 * @author ASUS
 */
public class Iseng4 {    
    
    public static void main(String[] args) {
        int noodd = -1;
        int odd = 0;
        Scanner sc = new Scanner(System.in);
        
        System.out.print("Input number of elements: ");
        int num = sc.nextInt();
        int A[] = new int[num];
        
        System.out.println("Input number: ");
        for(int i = 0; i < num; i++){
            A[i] = sc.nextInt();
        }
        
        System.out.println("\n");
        
        for(int i = 0; i < 5; i++){
            try{
                if(A[i] % 2 == 1){
                    odd = odd + A[i];                    
                    System.out.println(A[i]);
                }
            }catch(Exception EOException){
                break;
            }
        }
        
        if(odd != 0){
            System.out.println("output = " + odd);
        }else{
            System.out.println("output = " + noodd);
        }
    }
    
}

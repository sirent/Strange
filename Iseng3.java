/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package iseng;

/**
 *
 * @author ASUS
 */
public class Iseng3 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        int i = 1;
        int j = 0;
        int k = 0;
        int odd = 0;
        int even = 0;
        
        while(i < 10){              // limited from 1 to 10
            if(i % 2 == 0){         // check odd number
                odd = odd + 1;
                j = j + i;
            }else if(i % 2 == 1){   // check even number
                even = even + 1;
                k = k + i;
            }
            i++;
        }
        System.out.println("Sum odd number: " + j);
        System.out.println("odd number counter: " + odd);
        System.out.println("\nSum even number: " + k);
        System.out.println("even number counter: " + even);
    }
    
}

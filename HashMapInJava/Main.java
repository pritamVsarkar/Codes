import java.util.HashMap;
import java.util.Map;
public class Main{
    public static void main(String args[]){
        //HashMap<key_field,value_field> NamE_of_HashMap = new HashMap<>(); 
        HashMap<String,Integer> pritam=new HashMap<>(); //pritam new HashMap object
        print(pritam);
        pritam.put("Pritam",25); //NamE_of_HashMap.put(key,val) -->store the key value pair
        pritam.put("Eshani",25);
        pritam.put("Rahul",25);
        pritam.put("Pujki",24);
        print(pritam);
        System.out.println(pritam.size());
        if(pritam.containsKey("Eshani")){// NamE_of_HashMap.containsKey(key)-->whether ke present or not
            Integer a=pritam.get("Eshani");//NamE_of_HashMap.get(key) -->get the corresponding value
            System.out.println("Eshani :"+a);
        }
        pritam.clear(); //NamE_of_HashMap.clear()-->make the hash map empty
        print(pritam); 
    }
    public static void print(Map<String, Integer> pritam)  
    { 
        if (pritam.isEmpty())  //NamE_of_HashMap.isEmpty() -->whether the has map empty or not
        { 
            System.out.println("pritam is empty"); 
        } 
          
        else
        { 
            System.out.println(pritam); 
        } 
    } 
}

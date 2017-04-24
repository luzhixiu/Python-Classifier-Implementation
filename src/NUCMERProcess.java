package nucmer.process;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

class Contig {

     String S1, S2,E1, E2, LEN1, LEN2, IDY, LENR, LENQ, COVR, COVQ, TAG1,TAG2="";

    public Contig(String S_1,String E_1,String S_2, String E_2, String LEN_1, String LEN_2, String IDY_, String LEN_R, String LEN_Q, String COV_R, String COV_Q, String TAG_1,String TAG_2) {
        //System.out.println(LEN_2);
        LEN2="";
        S1 = S_1;
        S2 = S_2;
        E1 = E_1;
        E2 = E_2;
        LEN1 = LEN_1;
        LEN2 = LEN_2;
        IDY = IDY_;
        LENR = LEN_R;
        LENQ = LEN_Q;
        COVR = COV_R;
        COVQ = COV_Q;
        TAG1 = TAG_1;
        TAG2 = TAG_2;
    }

    static Comparator<Contig> LEN2Comparator() {
        return new Comparator<Contig>(){
            @Override
            public int compare(Contig c1, Contig c2) {
                return (int)(Double.parseDouble(c1.LEN2)-Double.parseDouble(c2.LEN2));
            }
           
        };
    }
    
//    int QLength
    
        static Comparator<Contig> IDYComparator() {
        return new Comparator<Contig>(){
            @Override
            public int compare(Contig c1, Contig c2) {
                return (int)(Double.parseDouble(c1.IDY)-Double.parseDouble(c2.IDY));
            }
           
        };
    }
//        static void print(){
//            System.out.println("===="+S1+" "+S2+" "+E1+" "+2+" "+LEN1+" "+LEN2+" "+IDY+" "+LENR+" "+LENQ+" "+COVR+" "+COVQ+" "+TAG1+" "+TAG2);
//        
//        }
        
    
    
    

}

public class NUCMERProcess {
        
    public static void main(String[] args) throws FileNotFoundException {
        Boolean start=false;
        ArrayList<Contig> ai = new ArrayList();
        Scanner sc = new Scanner(new File("B5.txt"));
        int cnt=0;
        Contig ct=null;
        while (sc.hasNext()) {           
            cnt++;
            String line;
            if(!start){
                line=sc.nextLine();
                if(line.contains("=="))start=true;
            continue;
            } 
            //if(cnt>620)break;
            line = sc.nextLine();
            
            String roughlist[]=lineSplit(line);
            String list[]=removeEmpty(roughlist);
//            for(int i=0;i<list.length;i++){
//                list[i]=list[i].trim();
//            }
           //  System.out.println(Arrays.toString(list));
           // System.out.println(list[5]);
            ct=new Contig(list[0],list[1],list[2],list[3],list[4],list[5],list[6],list[7],list[8],list[9],list[10],list[11],list[12]);
            ai.add(ct);
            }
            //System.out.println(ai.size());
            Collections.sort(ai,Contig.LEN2Comparator().reversed());//sort by LEN2
            Set<String> Tag2Set=new HashSet(); 
            ArrayList<Contig> distinct=new ArrayList();//this arraylistremoves the duplicate
            for(Contig c:ai){
               if(Tag2Set.add(c.TAG2))distinct.add(c);
            }
            System.out.println("This is the result with no filter");
            System.out.println("number of single hit: "+distinct.size());
            double sum=0;

            for(Contig c:distinct)sum+=Double.parseDouble(c.LEN2);
            System.out.println("sum of the length: "+sum);
            double IDYsum=0;
            
            for(Contig c:distinct)IDYsum+=Double.parseDouble(c.IDY);
            System.out.println("average identity: "+IDYsum/distinct.size());
            //==================================================================
            System.out.println("");
            System.out.println("This is the result with 90% identity and 70% Coverage");
            ArrayList<Contig> filter1=new ArrayList();
            for(Contig c:distinct){
                if(toDouble(c.IDY)>=90&&toDouble(c.COVQ)>=70){filter1.add(c);}
            }
            
            System.out.println("number of single hit: "+filter1.size());
            double sum1=0;

            for(Contig c:filter1)sum1+=Double.parseDouble(c.LEN2);
            System.out.println("sum of the length "+sum1);
            double IDYsum1=0;
            
            for(Contig c:filter1)IDYsum1+=Double.parseDouble(c.IDY);
            System.out.println("average identity: "+IDYsum1/filter1.size());
            //==================================================================
            System.out.println("");
            System.out.println("This is the result with 95% identity and 70% Coverage");
            ArrayList<Contig> filter2=new ArrayList();
            for(Contig c:distinct){
                if(toDouble(c.IDY)>=95&&toDouble(c.COVQ)>=70){filter2.add(c);}
            }
            
            System.out.println("number of single hit: "+filter2.size());
            double sum2=0;

            for(Contig c:filter2)sum2+=Double.parseDouble(c.LEN2);
            System.out.println("sum of the length "+sum2);
            double IDYsum2=0;
            
            for(Contig c:filter2)IDYsum2+=Double.parseDouble(c.IDY);
            System.out.println("average identity: "+IDYsum2/filter2.size());
         
            
            
            
            
            
            

    }

    static public String[] lineSplit(String line) {
        ArrayList<String> ai = new ArrayList();
        String list[] = line.split("\\|");
        for (int i = 0; i < list.length; i++) {
            String s = list[i];
            String hunks[] = s.split("\\s+");
            for (int j = 0; j < hunks.length; j++) {
                ai.add(hunks[j]);
                
            }
        }
        return ai.toArray(new String[0]);
    }

    static public String[] removeEmpty(String[] list){
        ArrayList<String> ai=new ArrayList();
        for(int i=0;i<list.length;i++){
            if(list[i].length()==0)continue;
            ai.add(list[i]);
        }
        return ai.toArray(new String[0]);     
    }
    
   
    static void printContig(Contig ct){
        System.out.println(ct.LEN2);
    }
    
    static double toDouble(String s){
    return Double.parseDouble(s);
    }
    
    
}

package gmapsprocessing;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

class Contig {

    int QL;
    String QN;
    int row;
    public Contig(String QName, int QLength,int rowNumber) {
        QL = QLength;
        QN = QName;
        row=rowNumber;
    }

    static Comparator<Contig> LEN2Comparator() {
        return new Comparator<Contig>() {
            @Override
            public int compare(Contig c1, Contig c2) {
                return (int) (c1.QL - c2.QL);
            }

        };
    }

    
    
}




public class GmapsProcessing {
    static File f;

    public static void main(String[] args) throws FileNotFoundException, IOException {
        f=new File("B5_EST.psl");
        Scanner sc = new Scanner(f);
        int cnt = 0;
        ArrayList<Contig> contigs=new ArrayList();
        while (sc.hasNext()) {
            
            String line=sc.nextLine();
            String roughlist[]=lineSplit(line);
            
        String list[]=removeEmpty(roughlist);
//            for (int i = 0; i < list.length; i++) {
//                System.out.print(list[i] + "         ");
//            }
//        System.out.println(list[11]);
          Contig ct=new Contig(list[9],Integer.parseInt(list[11]),cnt);  
          contigs.add(ct);
          cnt++;
        }
        
        
        System.out.println("Cnt:"+cnt);
        
        //sort the queries
        Collections.sort(contigs,Contig.LEN2Comparator().reversed());
        
//        for(int i=0;i<contigs.size();i++){
//            System.out.println(contigs.get(i).QN+"   "+contigs.get(i).QL+" "+contigs.get(i).row);
//            
//        }
        
        //remove duplicates based on names
        Set<String> QNSet=new HashSet(); 
        
        
        
        ArrayList<Contig> distinct = new ArrayList();//this arraylistremoves the duplicate
        int sum=0;
        for (Contig c : contigs) {
             if(QNSet.add(c.QN))distinct.add(c);
             sum+=c.QL;
        }
//        for (int i = 0; i < distinct.size(); i++) {
//            System.out.println(distinct.get(i).QN + "   " + contigs.get(i).QL);
//        }
//
//        
//        
        System.out.println("Number of single hit: "+distinct.size());
        System.out.println("Total Hit Length: "+sum);
       // writeToFile(distinct,(f.getName()+"Unfiltered"));
        
        }
        
        
        
        
        
        




    
    
    
    
        static public String[] lineSplit(String line) {
        ArrayList<String> ai = new ArrayList();
        String list[] = line.split("  ");
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
        
        static public void writeToFile(ArrayList<Contig> list,String FName) throws IOException{
            File output=new File(FName);
            BufferedWriter bw=new BufferedWriter(new FileWriter(output));
            Set<Integer> validRows=new HashSet();
            for(int i=0;i<list.size();i++){
                validRows.add(list.get(i).row);
                }
            Scanner sc=new Scanner(f);
            int cnt=0;
            int writenCNT=0;
            while(sc.hasNext()){
                String line=sc.nextLine();
                if(validRows.contains(cnt)){bw.write(line);
                    System.out.println("write row "+cnt);
                    writenCNT++;
                    }
                cnt++;
            }
            System.out.println(writenCNT +" lines were wrote to the file"+FName);
            bw.close();
        
        }
        
    
    
    
    
    

}
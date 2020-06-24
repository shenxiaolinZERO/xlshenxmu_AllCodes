package librec.main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

public class Test {

	public static void main(String[] args) {
        Map<String, Double> map = new TreeMap<String, Double>();
        Comparator<Entry<String, Double>> comp = new Comparator<Map.Entry<String,Double>>() {
            //��������
            public int compare(Entry<String, Double> o1,
                    Entry<String, Double> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        };
        map.put("a",  (double) 4);
        map.put("c", (double) 3);
        map.put("d", (double) 4.8);
        map.put("b", (double) 2.6);
        
        //���ｫmap.entrySet()ת����list
        List<Entry<String, Double>> list = new ArrayList<Map.Entry<String,Double>>(map.entrySet());
        //Ȼ��ͨ���Ƚ�����ʵ������
        Collections.sort(list,comp);
        
        for(Map.Entry<String,Double> mapping:list){ 
               System.out.println(mapping.getKey()+":"+mapping.getValue()); 
          } 
       
    }

}

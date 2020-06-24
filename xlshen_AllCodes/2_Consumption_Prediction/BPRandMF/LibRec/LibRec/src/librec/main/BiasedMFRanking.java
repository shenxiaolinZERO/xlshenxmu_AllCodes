// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;

import librec.util.FileIO;
import librec.util.Logs;
import librec.util.Strings;
import librec.util.Systems;

/**
 * A demo created for the UMAP'15 demo session, could be useful for other users.
 * 
 * @author Guo Guibing
 *
 */
public class BiasedMFRanking {
	public static  String trainFile=null,testFile=null;
	
	/**
	 * 每个session对应的items的排序列表
	 */
	public static Map<String,List<String>> recommendMap = new LinkedHashMap<String,List<String>>();
	/**
	 * 每个session对应的每个item的分数
	 * key:sessionId
	 * value:  map:key为itemId,value为分数
	 * 在Recommender.java中会计算分数并赋值给这个变量
	 */
	public static Map<String,TreeMap<String,Double>> ratingMap = new LinkedHashMap<String,TreeMap<String,Double>>();
	
	public static void main(String[] args) {
		try {
			new BiasedMFRanking().execute(args);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void execute(String[] args) throws Exception {

		// config logger
		String dirPath = FileIO.makeDirPath("demo");
		Logs.config(dirPath + "log4j.xml", true);

		// set the folder path for configuration files
		String configDirPath = FileIO.makeDirPath(dirPath, "config");

		// prepare candidate options
		List<String> candOptions = new ArrayList<>();
		candOptions.add("General Usage:");
		candOptions.add(" 0: the format of rating prediction results;");
		candOptions.add(" 1: the format of item recommendation results;");
		candOptions.add(" 2: run an algorithm by name [Input: 2 algoName];");
		candOptions.add(" 3: help & about this demo;");
		candOptions.add("-1: quit the demo!");
		candOptions.add("");
		candOptions.add("Part I: baselines");
		candOptions.add("10: Global Average;   11: User Average;  12: Item Average;");
		candOptions.add("13: Most Popularity;  14: User Cluster;  15: Item Cluster;");
		candOptions.add("16: Association Rule; 17: Non-neg MF;    18: Slope One;");
		candOptions.add("");
		candOptions.add("Part II: rating prediction");
		candOptions.add("20: UserKNN;\t 21: ItemKNN; \t 22: TrustSVD; ");
		candOptions.add("23: RegSVD; \t 24: BiasedMF;\t 25: SVD++; ");
		candOptions.add("");
		candOptions.add("Part III: item recommendation");
		candOptions.add("30: LDA;    \t 31: BPR;     \t 32: FISM; ");
		candOptions.add("33: WRMF;   \t 34: SLIM;    \t 35: RankALS. ");

		int option = 0;
		boolean flag = false;
		Scanner reader = new Scanner(System.in);
		String configFile = "librec.conf";
		do {
			Logs.debug(Strings.toSection(candOptions));
			System.out.print("Please choose your command id: ");
			option = 24;//reader.nextInt();

			// print an empty line
			Logs.debug();
			flag = false;

			// get algorithm-specific configuration file
			switch (option) {
			case 10:
				configFile = "GlobalAvg.conf";
				break;
			case 11:
				configFile = "UserAvg.conf";
				break;
			case 12:
				configFile = "ItemAvg.conf";
				break;
			case 13:
				configFile = "MostPop.conf";
				break;
			case 14:
				configFile = "UserCluster.conf";
				break;
			case 15:
				configFile = "ItemCluster.conf";
				break;
			case 16:
				configFile = "AR.conf";
				break;
			case 17:
				configFile = "NMF.conf";
				break;
			case 18:
				configFile = "SlopeOne.conf";
				break;
			case 20:
				configFile = "UserKNN.conf";
				break;
			case 21:
				configFile = "ItemKNN.conf";
				break;
			case 22:
				configFile = "TrustSVD.conf";
				break;
			case 23:
				configFile = "RegSVD.conf";
				break;
			case 24:
				configFile = "BiasedMF.conf";
				break;
			case 25:
				configFile = "SVD++.conf";
				break;
			case 30:
				configFile = "LDA.conf";
				break;
			case 31:
				configFile = "BPR.conf";
				break;
			case 32:
				configFile = "FISM.conf";
				break;
			case 33:
				configFile = "WRMF.conf";
				break;
			case 34:
				configFile = "SLIM.conf";
				break;
			case 35:
				configFile = "RankALS.conf";
				break;
			case -1:
				flag = true;
				break;
			case 0:
				Logs.debug("Prediction results: MAE, RMSE, NMAE, rMAE, rRMSE, MPE, <configuration>, training time, test time\n");
				Systems.pause();
				continue;
			case 1:
				Logs.debug("Ranking results: Prec@5, Prec@10, Recall@5, Recall@10, AUC, MAP, NDCG, MRR, <configuration>, training time, test time\n");
				Systems.pause();
				continue;
			case 2:
				// System.out.print("Please input the method name: ");
				String algoName = reader.next().trim();
				configFile = algoName + ".conf";
				break;
			case 3:
				StringBuilder about = new StringBuilder();
				about.append("About. This demo was created by Guo Guibing, the author of the LibRec library.\n")
						.append("It is based on LibRec-v1.3 (http://www.librec.net/). Although initially designed\n")
						.append("for a demo session at UMAP'15, it may be useful for those who want to take a \n")
						.append("quick trial of LibRec. Source code: https://github.com/guoguibing/librec.\n\n")
						.append("Usage. To run a predefined recommender, simply choose a recommender id.\n")
						.append("To run a customized recommender, give the input '2 algoName' (e.g., '2 RegSVD').\n")
						.append("For case 2, make sure you have a configuration file named by 'algoName.conf'\n");

				Logs.debug(about.toString());
				Systems.pause();
				continue;
			default:
				Logs.error("Wrong input id!\n");
				Systems.pause();
				continue;
			}

			if (flag)
				break;
			String fileDir;
			
			fileDir = ".\\data\\ratings\\";
			Map<Integer,List<Integer>> map = new HashMap<Integer,List<Integer>>();
		
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("MFResult.txt")));
			map.put(1,Arrays.asList(2,8,18,19,22,28,36,40,44,49));
			map.put(2,Arrays.asList(3,7,12,18,21,35,37,44,46,49));
			map.put(3,Arrays.asList(1,6,8,9,11,24,33,45,47,50));
			map.put(4,Arrays.asList(3,4,5,18,20,25,29,39,45,49));
			map.put(5,Arrays.asList(18,19,25,28,34,38,39,40,45,47));
			map.put(6,Arrays.asList(5,6,9,12,24,28,33,39,40,49));
			
			for(int i = 1;i<=6;i++)
			{	
				
				List<Integer> dataList = map.get(i);
				for(int j:dataList)
				{
					testFile = i+"_";
					trainFile = i+"_";
					recommendMap = new LinkedHashMap<String,List<String>>();
					ratingMap = new LinkedHashMap<String,TreeMap<String,Double>>();
					
					testFile = new File(fileDir).getCanonicalPath()+"\\"+testFile+(j+"_te.txt");
					trainFile = new File(fileDir).getCanonicalPath()+"\\"+trainFile+(j+"_tr.txt");
				
					// run algorithm
					LibRec librec = new LibRec();
					librec.setConfigFiles(configDirPath + configFile);
					librec.execute(args);
					
					/**
					 * ratingMap转为recommendMap
					 */
					transformRatingToRank();
				
					Map<String,Set<String>> sessionBuyItemMap = readBuyItem(testFile);
					
					float p ;
					if(i==1)
						p= calcPrecisionAt1(sessionBuyItemMap);
					else if(i==2)
						p= calcPrecisionAt2(sessionBuyItemMap);
					else if(i==3)
						p= calcPrecisionAt3(sessionBuyItemMap);
					else if(i==4)
						p= calcPrecisionAt4(sessionBuyItemMap);
					else if(i==5)
						p= calcPrecisionAt5(sessionBuyItemMap);
					else//if(i==6)
						p= calcPrecisionAt6(sessionBuyItemMap);
					float mrr = calcMRR(sessionBuyItemMap);
					System.out.println(p+" "+mrr);
					writer.write(i+"_"+j+"\t"+p+"\t"+mrr+"\r\n");
					writer.flush();
					//break;
				}
				//break;
			}
			// await next command
			writer.close();
			Logs.debug();
			Systems.pause();

		} while (option != -1);
		reader.close();

		Logs.debug("Thanks for trying out LibRec! See you again!");
	}

	private void transformRatingToRank() {
		Comparator<Entry<String, Double>> comp = new Comparator<Map.Entry<String,Double>>() {
            //升序排序
            public int compare(Entry<String, Double> o1,
                    Entry<String, Double> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        };
        for(String key:BiasedMFRanking.ratingMap.keySet())
        {
        	TreeMap<String,Double> map = BiasedMFRanking.ratingMap.get(key);
        	//这里将map.entrySet()转换成list
            List<Entry<String, Double>> list = new ArrayList<Map.Entry<String,Double>>(map.entrySet());
            //然后通过比较器来实现排序
            Collections.sort(list,comp);
            
            List<String> l = new ArrayList<String>();
            for(Map.Entry<String,Double> mapping:list){ 
                  // System.out.println(mapping.getKey()+":"+mapping.getValue()); 
            	l.add(mapping.getKey());
            } 
            BiasedMFRanking.recommendMap.put(key, l);
        }
		
	}

	private void evaluate(Map<String, Set<String>> sessionBuyItemMap) {
		
	}

	private float calcMRR(Map<String, Set<String>> sessionBuyItemMap) {
		float mrr = 0;
		List<String> l;
		for(String key:BiasedMFRanking.recommendMap.keySet())
		{
			l = BiasedMFRanking.recommendMap.get(key);
			Set<String> buyItemSet = sessionBuyItemMap.get(key);
			for(int i = 0;i<l.size();i++)
			{
				if(buyItemSet.contains(l.get(i)))
				{
					mrr += (1.0/(i+1));
					break;
				}
			}
		}
		return mrr/(BiasedMFRanking.recommendMap.keySet().size());
	}

	private float calcPrecisionAt1(Map<String, Set<String>> sessionBuyItemMap) {
		
		float p = 0;
		List<String> l;
		for(String key:BiasedMFRanking.recommendMap.keySet())
		{
			l = BiasedMFRanking.recommendMap.get(key);
			//System.out.println(key+" "+sessionBuyItemMap.containsKey(key)+" "+l.size());
			Set<String> buyItemSet = sessionBuyItemMap.get(key);
			if(buyItemSet.contains(l.get(0)))
			{
				p += 1.0;
			}
		}
		return p/(BiasedMFRanking.recommendMap.keySet().size());
	}
	private float calcPrecisionAt2(Map<String, Set<String>> sessionBuyItemMap) {
		
		float p = 0;
		List<String> l;
		for(String key:BiasedMFRanking.recommendMap.keySet())
		{
			l = BiasedMFRanking.recommendMap.get(key);
			Set<String> buyItemSet = sessionBuyItemMap.get(key);
			
			if(buyItemSet.contains(l.get(0)))
			{
				p += 1.0/2.0;
			}
			if(buyItemSet.contains(l.get(1)))
			{
				p += 1.0/2.0;
			}
		}
		return p/(BiasedMFRanking.recommendMap.keySet().size());
	}
	private float calcPrecisionAt3(Map<String, Set<String>> sessionBuyItemMap) {
		
		float p = 0;
		List<String> l;
		for(String key:BiasedMFRanking.recommendMap.keySet())
		{
			l = BiasedMFRanking.recommendMap.get(key);
			Set<String> buyItemSet = sessionBuyItemMap.get(key);
			
			if(buyItemSet.contains(l.get(0)))
			{
				p += 1.0/3.0;
			}
			if(buyItemSet.contains(l.get(1)))
			{
				p += 1.0/3.0;
			}
			if(buyItemSet.contains(l.get(2)))
			{
				p += 1.0/3.0;
			}
		}
		return p/(BiasedMFRanking.recommendMap.keySet().size());
	}
	private float calcPrecisionAt4(Map<String, Set<String>> sessionBuyItemMap) {
		
		float p = 0;
		List<String> l;
		for(String key:BiasedMFRanking.recommendMap.keySet())
		{
			l = BiasedMFRanking.recommendMap.get(key);
			Set<String> buyItemSet = sessionBuyItemMap.get(key);
			
			if(buyItemSet.contains(l.get(0)))
			{
				p += 1.0/4.0;
			}
			if(buyItemSet.contains(l.get(1)))
			{
				p += 1.0/4.0;
			}
			if(buyItemSet.contains(l.get(2)))
			{
				p += 1.0/4.0;
			}
			if(buyItemSet.contains(l.get(3)))
			{
				p += 1.0/4.0;
			}
		}
		return p/(BiasedMFRanking.recommendMap.keySet().size());
	}
	private float calcPrecisionAt5(Map<String, Set<String>> sessionBuyItemMap) {
		
		float p = 0;
		List<String> l;
		for(String key:BiasedMFRanking.recommendMap.keySet())
		{
			l = BiasedMFRanking.recommendMap.get(key);
			Set<String> buyItemSet = sessionBuyItemMap.get(key);
			
			if(buyItemSet.contains(l.get(0)))
			{
				p += 1.0/5.0;
			}
			if(buyItemSet.contains(l.get(1)))
			{
				p += 1.0/5.0;
			}
			if(buyItemSet.contains(l.get(2)))
			{
				p += 1.0/5.0;
			}
			if(buyItemSet.contains(l.get(3)))
			{
				p += 1.0/5.0;
			}
			if(buyItemSet.contains(l.get(4)))
			{
				p += 1.0/5.0;
			}
		}
		return p/(BiasedMFRanking.recommendMap.keySet().size());
	}
	private float calcPrecisionAt6(Map<String, Set<String>> sessionBuyItemMap) {
		
		float p = 0;
		List<String> l;
		for(String key:BiasedMFRanking.recommendMap.keySet())
		{
			l = BiasedMFRanking.recommendMap.get(key);
			Set<String> buyItemSet = sessionBuyItemMap.get(key);
			
			if(buyItemSet.contains(l.get(0)))
			{
				p += 1.0/6.0;
			}
			if(buyItemSet.contains(l.get(1)))
			{
				p += 1.0/6.0;
			}
			if(buyItemSet.contains(l.get(2)))
			{
				p += 1.0/6.0;
			}
			if(buyItemSet.contains(l.get(3)))
			{
				p += 1.0/6.0;
			}
			if(buyItemSet.contains(l.get(4)))
			{
				p += 1.0/6.0;
			}
			if(buyItemSet.contains(l.get(5)))
			{
				p += 1.0/6.0;
			}
		}
		return p/(BiasedMFRanking.recommendMap.keySet().size());
	}
	private Map<String, Set<String>> readBuyItem(String testFile) {
		HashMap<String, Set<String>> map = new HashMap<String,Set<String>>();
		String line="";
		String s[];
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(testFile)));
			while(true)
			{
				try {
					line=reader.readLine();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				if(line==null)break;
				line = line.trim();
				if(line.length()==0)break;
				
				s = line.split("\\s+");
				
				if(Float.parseFloat(s[2])==1)
				{
					if(map.containsKey(s[0]))
					{
						map.get(s[0]).add(s[1]);
					}
					else
					{
						Set<String> set = new HashSet<String>();
						set.add(s[1]);
						map.put(s[0], set);
					}
				}
				
				
			}

			try {
				reader.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return map;
	}
}

package prea.main;

public class ThreadTest {
	public static void main(String[] args) throws InterruptedException {
		int THREAD_COUNT = 4;
		
		// Multi-thread version
		long parStart = System.currentTimeMillis();
		
		ThreadClass[] threadArray = new ThreadClass[THREAD_COUNT];
		for (int iter = 0; iter < THREAD_COUNT; iter++) {
			threadArray[iter] = new ThreadClass(iter+1);
			threadArray[iter].start();
		}
        
		for (int iter = 0; iter < THREAD_COUNT; iter++) {
			threadArray[iter].join();
		}
        
        long parEnd = System.currentTimeMillis();
		System.out.println ("ParTime: " + (parEnd - parStart));
		
		for (int iter = 0; iter < THREAD_COUNT; iter++) {
			System.out.println(threadArray[iter].msg);
		}
    }
}

class ThreadClass extends Thread {
	int length;
	String msg;

	ThreadClass(int len) {
		length = len;
		msg = "";
	}
	
	public void run() {
		for (int i = 0; i < length; i++) {
			char b = (char) (Math.random() * 26);
			b += 'a';
			msg += b;
		}
		//System.out.println(msg);
	}
}
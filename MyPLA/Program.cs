/*************************************************************** 
 * 依據信用條件評估被評比人是否可核發信用卡
 * 透過 PLA 算法找出最佳權重與閥值，
 * 預測新的被評比人資料是否符合預期值
 * 
 * // 信用評比條件 //
 * -------------------------------------------------------------
   | 工作(X1) |	收入(X2)    | 信用歷史紀錄(X3) | 貸款(X4)| 預期值
   -------------------------------------------------------------
   | 無=1     | 1萬以下 = 1 |   差=1           |  無=1   |
   | 有=2	  | 1-4萬   = 2 |  尚可=2          | 有=2    |
   |          | 4-8萬   = 3 |  優=3	           |         |
   |          | 8萬以上 = 4 |                  |         |
****************************************************************/

using System;

namespace MyPLA
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                // 輸入
                Console.WriteLine("\nBegin Perceptron demo\n");
                int[][] trainingData = new int[10][];
                // 最後一個值是期望值
                trainingData[0] = new int[] { 2, 3, 3, 2, 1 };  
                trainingData[1] = new int[] { 1, 1, 2, 2, 0 };  
                trainingData[2] = new int[] { 2, 1, 3, 1, 1 };  
                trainingData[3] = new int[] { 1, 2, 3, 2, 0 };  
                trainingData[4] = new int[] { 1, 4, 2, 2, 1 };
                trainingData[5] = new int[] { 1, 1, 1, 1, 0 };
                trainingData[6] = new int[] { 2, 4, 3, 1, 1 };
                trainingData[7] = new int[] { 2, 3, 2, 1, 1 };
                trainingData[8] = new int[] { 2, 2, 2, 2, 1 };
                trainingData[9] = new int[] { 1, 1, 1, 2, 0 };

                ShowData(trainingData[0]);
                ShowData(trainingData[1]);
                ShowData(trainingData[2]);
                ShowData(trainingData[3]);
                ShowData(trainingData[4]);
                ShowData(trainingData[5]);
                ShowData(trainingData[6]);
                ShowData(trainingData[7]);
                ShowData(trainingData[8]);
                ShowData(trainingData[9]);

                Console.WriteLine("\n\nFinding best weights and bias");
                int maxEpochs = 1000;
                double alpha = 0.075; //學習率
                double targetError = 0.0;
                double bestBias = 0.0; // 閥值
                double[] bestWeights = FindBestWeights(trainingData, maxEpochs, alpha, targetError, out bestBias);
                Console.WriteLine("\nTraining complete");

                double totalError = TotalError(trainingData, bestWeights, bestBias);
                Console.WriteLine("\nAfter training total error = " + totalError.ToString("F4"));

                // 丟入新資料來測試
                //int[] unknown = new int[] { 1, 1, 1, 2, 0 };
                //int[] unknown = new int[] { 2, 3, 3, 1, 1 };
                int[] unknown = new int[] { 1, 2, 2, 2, 0 };

                Console.WriteLine("\n丟入新資料來測試\n");
                ShowData(unknown);

                int prediction = Predict(unknown, bestWeights, bestBias);  // perform the classification
                string s0 = "預測錯誤";
                string s1 = "預測正確";
                if (prediction == 0) Console.WriteLine(s0); else Console.WriteLine(s1);

                Console.WriteLine("\nEnd Perceptron demo\n");

            }
            catch (Exception ex)
            {
            }

            Console.Read();
 
        }

        // 找出最佳權重群
        public static double[] FindBestWeights(int[][] trainingData, int maxEpochs, double alpha, double targetError, out double bestBias)
        {
            int dim = trainingData[0].Length - 1; // 扣掉最後一個值(期望值)
            double[] weights = new double[dim];  // 初始化所有權重為 0
            double bias = 0.05; // 初始化閥值
            double totalError = double.MaxValue; // 初始化錯誤值總量
            int epoch = 0;

            while (epoch < maxEpochs && totalError > targetError)
            //while (epoch < maxEpochs)
            {
                for (int i = 0; i < trainingData.Length; ++i)  // each training vector
                {
                    // 每一期望值
                    int desired = trainingData[i][trainingData[i].Length - 1];  // last bit
                    // 每一輸出值
                    int output = ComputeOutput(trainingData[i], weights, bias);

                    int delta = desired - output;  // -1 (if output too large), 0 (output correct), or +1 (output too small)

                    for (int j = 0; j < weights.Length; ++j)
                        weights[j] = weights[j] + (alpha * delta * trainingData[i][j]);  // 更新每一權重

                    bias = bias + (alpha * delta); // 更新閥值
                }

                totalError = TotalError(trainingData, weights, bias);  // rescans; could do in for loop
                ++epoch;
            }
            bestBias = bias;
            return weights;
        }

        // 計算總輸出值
        public static int ComputeOutput(int[] trainVector, double[] weights, double bias)
        {
            double dotP = 0.0;
            for (int j = 0; j < trainVector.Length - 1; ++j)  // not last bit which is the desired
                dotP += (trainVector[j] * weights[j]);
            dotP += bias;
            return StepFunction(dotP);
        }

        // 計算總錯誤值
        public static double TotalError(int[][] trainingData, double[] weights, double bias)
        {
            double sum = 0.0;
            for (int i = 0; i < trainingData.Length; ++i)
            {
                int desired = trainingData[i][trainingData[i].Length - 1];
                int output = ComputeOutput(trainingData[i], weights, bias);
                sum += (desired - output) * (desired - output);  // equivalent to Abs(desired - output) in this case
            }
            return 0.5 * sum;
        }

        // 預測
        public static int Predict(int[] dataVector, double[] bestWeights, double bestBias)
        {
            int res = 0;
            double dotP = 0.0;
            try
            {
                for (int j = 0; j < dataVector.Length -1 ; ++j)  // all bits
                    dotP += (dataVector[j] * bestWeights[j]);
                dotP += bestBias;
            }
            catch(Exception ex)
            {

            }

            var predictResult = StepFunction(dotP);
            //如果預測數值=期望值
            if(predictResult == dataVector[4])
            {
                res = 1;
            }
            return res;
        }

        // 步階函數 回傳 0 或 1
        public static int StepFunction(double x)
        {
            if (x > 0.5) return 1; else return 0;
        }

        // 印出資料
        public static void ShowData(int[] data)
        {
            for (int i = 0; i < 5; ++i)  // hard-coded
            {

                Console.Write(data[i]+",");
            }
            Console.WriteLine("");
        }
    }
}

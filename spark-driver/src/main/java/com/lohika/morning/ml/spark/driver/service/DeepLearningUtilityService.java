package com.lohika.morning.ml.spark.driver.service;

import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
// https://github.com/deeplearning4j/dl4j-spark-cdh5-examples/issues/18
public class DeepLearningUtilityService {

    private static int batchSizePerWorker = 64;

    @Value("${mnist.deep.learning.parquet.file.path}")
    private String mnistDeepLearningParquetFilePath;

    @Autowired
    private SparkSession sparkSession;

    public void saveMNISTDataset() {
//        try {
//            DataSetIterator iter = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
//            List<DataSet> list = new ArrayList<>();
//            while (iter.hasNext()) {
//                list.add(iter.next());
//            }
//
//            JavaRDD<DataSet> rdd = new JavaSparkContext(sparkSession.sparkContext()).parallelize(list);
//            URI outputURI = new URI(mnistDeepLearningParquetFilePath);
//            rdd.foreachPartition(new DataSetExportFunction(outputURI));
//        } catch (Exception exception) {
//            System.out.println("Something went wrong during saving of MNIST dataset" + exception.getMessage());
//        }
    }

}

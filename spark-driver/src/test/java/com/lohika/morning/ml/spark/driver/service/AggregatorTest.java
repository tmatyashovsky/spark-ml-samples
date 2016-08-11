package com.lohika.morning.ml.spark.driver.service;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import org.junit.Test;
import org.springframework.beans.factory.annotation.Autowired;

public class AggregatorTest extends BaseTest {

    @Autowired
    private TextService textService;

    @Test
    public void shouldCalculateCorrectAvgAndCovariance() {
        JavaRDD<Row> data = new JavaSparkContext(getSparkSession().sparkContext())
            .parallelize(Arrays.asList(
                RowFactory.create("And nothing else matters", 0D, 1D, "And", Vectors.dense(1.0, 2.0, 3.0)),
                RowFactory.create("And nothing else matters", 0D, 1D, "nothing", Vectors.dense(1.0, 2.0, 3.0)),
                RowFactory.create("And nothing else matters", 0D, 1D, "else", Vectors.dense(4.0, 5.0, 6.0)),
                RowFactory.create("And nothing else matters", 0D, 1D, "matters", Vectors.dense(4.0, 8.0, 12.0)))
            );

            StructType schema = new StructType(new StructField[]{
                new StructField("value", DataTypes.StringType, false, Metadata.empty()),
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("words", DataTypes.StringType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });

        Dataset<Row> dataset = getSparkSession().createDataFrame(data, schema);

        Dataset<Row> averagesDataset = textService.getFeatures(dataset, 3, true);

        List<Row> averagesList = averagesDataset.collectAsList();
        assertEquals(1, averagesList.size());

        double[] averages = ((DenseVector) averagesList.get(0).getAs("averages")).values();
        assertArrayEquals(new double[]{2.25D, 3.75D, 6D}, averages, 0D);

        Dataset<Row> joined = dataset.join(averagesDataset, "id");

        Dataset<Row> variancesDataset = textService.getVariances(joined, 3);
        List<Row> variancesList = variancesDataset.collectAsList();
        assertEquals(1, variancesList.size());

        double[] variances = ((DenseVector) variancesList.get(0).getAs("variances")).values();
        assertArrayEquals(new double[]{3D, 4.5D, 6D, 4.5D, 8.25D, 12D, 6D, 12D, 18D}, variances, 0D);
    }

}

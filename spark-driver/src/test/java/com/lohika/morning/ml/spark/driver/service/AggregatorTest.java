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
                RowFactory.create(0D, 1D, Vectors.dense(1.0, 2.0, 3.0)),
                RowFactory.create(0D, 1D, Vectors.dense(4.0, 5.0, 6.0)),
                RowFactory.create(0D, 1D, Vectors.dense(4.0, 8.0, 12.0)))
            );

        StructType schema = new StructType(new StructField[]{
            new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("id", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("features", new VectorUDT(), false, Metadata.empty())
        });

        Dataset<Row> dataset = getSparkSession().createDataFrame(data, schema);

        textService.setVectorSize(3);
        Dataset<Row> averagesDataset = textService.getAverages(dataset);

        List<Row> averagesList = averagesDataset.collectAsList();
        assertEquals(1, averagesList.size());

        double[] averages = ((DenseVector) averagesList.get(0).getAs("averages")).values();
        assertArrayEquals(new double[]{3D, 5D, 7D}, averages, 0D);

        Dataset<Row> joined = dataset.join(averagesDataset, "id");

        Dataset<Row> variancesDataset = textService.getVariances(joined);
        List<Row> variancesList = variancesDataset.collectAsList();
        assertEquals(1, variancesList.size());

        double[] variances = ((DenseVector) variancesList.get(0).getAs("variances")).values();
        assertArrayEquals(new double[]{3D, 4.5D, 6D, 4.5D, 9D, 13.5D, 6D, 13.5D, 21D}, variances, 0D);
    }

}

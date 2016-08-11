package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.expressions.Aggregator;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

public class DenseVectorValuesElementsAverageAggregator extends Aggregator<Row, DoubleArrayAVGHolder, Row> {

    private final int vectorSize;
    private final boolean includeLabel;

    public DenseVectorValuesElementsAverageAggregator(int vectorSize, boolean includeLabel) {
        this.vectorSize = vectorSize;
        this.includeLabel = includeLabel;
    }

    @Override
    public DoubleArrayAVGHolder zero() {
        // NOTE: This can be easily implemented for unknown-sized rows input, if used with List instead of arrays
        // TODO: Check difference in performance for array-based and list-based implementations.
        return new DoubleArrayAVGHolder(vectorSize);
    }

    @Override
    public DoubleArrayAVGHolder reduce(DoubleArrayAVGHolder buffer, Row input) {
        // TODO: How should null-values be handled?
        if (input != null) {
            double[] averageSums = buffer.getAverageSums();
            int count = buffer.getCounts();

            DenseVector vector = (DenseVector) input.getAs("features");
            double[] vectorValues = vector.values();
            for (int i = 0, valuesLength = vectorValues.length; i < valuesLength; i++) {
                averageSums[i] = averageSums[i] + vectorValues[i];
            }

            buffer.setCounts(++count);

//            WrappedArray<WrappedArray<Double>> rows = ((WrappedArray<WrappedArray<Double>>)input.getAs("cov")).thisCollection();
//            double[][] covarianceSums = new double[vectorSize][vectorSize];
//
//            for (int i=0; i<vectorSize; i++) {
//                for (int j=0; j<vectorSize; j++) {
//                    covarianceSums[i][j] = rows.apply(i).apply(j);
//                }
//            }
//
//            buffer.setCovarianceSums(covarianceSums);

            if (includeLabel) {
                buffer.setLabel(input.getAs("label"));
            }
        }
        return buffer;
    }

    @Override
    public DoubleArrayAVGHolder merge(DoubleArrayAVGHolder buffer1, DoubleArrayAVGHolder buffer2) {
        int length = buffer1.getAverageSums().length;

        double[] buffer1Sums = buffer1.getAverageSums();
        double[] buffer2Sums = buffer2.getAverageSums();
        double[] sums = new double[length];
        for (int i = 0, sumsLength = buffer1Sums.length; i < sumsLength; i++) {
            sums[i] = buffer1Sums[i] + buffer2Sums[i];
        }

        int buffer1Counts = buffer1.getCounts();
        int buffer2Counts = buffer2.getCounts();
        int counts = buffer1Counts + buffer2Counts;

//        double[][] buffer1CovarianceSums = buffer1.getCovarianceSums();
//        double[][] buffer2CovarianceSums = buffer2.getCovarianceSums();
//        double[][] covarianceSums = new double[length][length];
//        for (int i = 0; i < vectorSize; i++) {
//            for (int j = 0; j < vectorSize; j++) {
//                covarianceSums[i][j] = buffer1CovarianceSums[i][j] + buffer2CovarianceSums[i][j];
//            }
//        }

        return new DoubleArrayAVGHolder(buffer1.getLabel() != null ? buffer1.getLabel() : buffer2.getLabel(),
                                        sums, counts, null);
    }

    @Override
    public Row finish(DoubleArrayAVGHolder buffer) {
        double[] sums = buffer.getAverageSums();
        int counts = buffer.getCounts();
        double[] averages = new double[sums.length];
        for (int i = 0, sumsLength = sums.length; i < sumsLength; i++) {
            averages[i] = sums[i] / counts;
        }

//        double[][] covarianceSums = buffer.getCovarianceSums();
//        double[] covariance = new double[vectorSize * vectorSize];
//
//        int z = 0;
//        for (int i = 0; i < vectorSize; i++) {
//            for (int j = 0; j < vectorSize; j++) {
//                covariance[z++] = covarianceSums[i][j]/ (counts);
//            }
//        }

        return RowFactory.create(new Object[]{buffer.getLabel(), new DenseVector(averages)});
    }

    @Override
    public Encoder<DoubleArrayAVGHolder> bufferEncoder() {
        return Encoders.bean(DoubleArrayAVGHolder.class);
    }

    @Override
    public Encoder<Row> outputEncoder() {
        // This should NOT be a RowEncoder, as it creates additional nested structure on top of the schema provided below (which is expected)
        // TODO: Find a way to pass Array directly (with Encoders.kryo it results in binary column).
        // NOTE: In spark scala api (as always), it is as easy as "Encoder[Array[Double]] = ExpressionEncoder()"..
        // ... but in java - https://issues.apache.org/jira/browse/SPARK-13128
        return RowEncoder.apply(DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField("label", DataTypes.DoubleType, true),
                DataTypes.createStructField("averages", new VectorUDT(), false)
//                DataTypes.createStructField("variances", new VectorUDT(), false)
        }));
    }
}
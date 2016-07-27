package com.lohika.morning.ml.spark.driver;

import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;

@Configuration
@PropertySource("classpath:spark.properties")
@ComponentScan("com.lohika.morning.ml.spark.driver.*")
public class SparkContextConfiguration {

    @Bean
    public SparkSession sparkSession() {
        return SparkSession
                .builder()
                .master(master)
                .appName(applicationName)
                .config("spark.cores.max", coresMax)
                .config("spark.driver.memory", driverMemory)
                .config("spark.executor.memory", executorMemory)
                .config("spark.serializer", serializer)
                .config("spark.kryoserializer.buffer.max", kryoserializerBufferMax)
                .config("spark.kryo.registrationRequired", "false")
                .config("spark.sql.shuffle.partitions", sqlShufflePartitions)
                .config("spark.default.parallelism", defaultParallelism)
                .getOrCreate();
    }

    @Value("${spark.master}")
    private String master;

    @Value("${spark.application-name}")
    private String applicationName;

    @Value("${spark.distributed-libraries}")
    private String[] distributedLibraries;

    @Value("${spark.cores.max}")
    private String coresMax;

    @Value("${spark.driver.memory}")
    private String driverMemory;

    @Value("${spark.executor.memory}")
    private String executorMemory;

    @Value("${spark.serializer}")
    private String serializer;

    @Value("${spark.sql.shuffle.partitions}")
    private String sqlShufflePartitions;

    @Value("${spark.default.parallelism}")
    private String defaultParallelism;

    @Value("${spark.kryoserializer.buffer.max}")
    private String kryoserializerBufferMax;

    @Bean
    // Static is extremely important here.
    // It should be created before @Configuration as it is also component.
    public static PropertySourcesPlaceholderConfigurer propertySourcesPlaceholderConfigurer() {
        return new PropertySourcesPlaceholderConfigurer();
    }

}




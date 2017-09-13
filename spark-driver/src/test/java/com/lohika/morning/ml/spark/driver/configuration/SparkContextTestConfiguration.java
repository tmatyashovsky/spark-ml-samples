package com.lohika.morning.ml.spark.driver.configuration;

import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

@Configuration
@PropertySource({"classpath:spark-test.properties", "classpath:application-test.properties"})
public class SparkContextTestConfiguration {

}

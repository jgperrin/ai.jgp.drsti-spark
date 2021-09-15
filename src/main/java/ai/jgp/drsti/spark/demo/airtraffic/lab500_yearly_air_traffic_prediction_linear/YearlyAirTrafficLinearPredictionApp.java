package ai.jgp.drsti.spark.demo.airtraffic.lab500_yearly_air_traffic_prediction_linear;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.year;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.RegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.jgp.drsti.spark.DrstiLineChart;
import ai.jgp.drsti.spark.DrstiUtils;
import ai.jgp.drsti.spark.utils.DataframeUtils;

/**
 * Projection to 2026 based on 2000-2019 data
 * 
 * @author jgp
 *
 */
public class YearlyAirTrafficLinearPredictionApp {
  private static Logger log =
      LoggerFactory.getLogger(YearlyAirTrafficLinearPredictionApp.class);

  public static void main(String[] args) {
    YearlyAirTrafficLinearPredictionApp app =
        new YearlyAirTrafficLinearPredictionApp();
    app.start();
  }

  /**
   * Real work goes here...
   */
  private boolean start() {
    log.debug("-> start()");
    long t0 = System.currentTimeMillis();

    // Creates a session on a local master
    SparkSession spark = SparkSession.builder()
        .appName("Projection to 2026 based on 2000-2019 data")
        .master("local[*]")
        .getOrCreate();

    long tc = System.currentTimeMillis();
    log.info("Spark master available in {} ms.", (tc - t0));
    t0 = tc;

    // Reading gold data from Delta
    Dataset<Row> goldDf = spark.read().format("delta")
        .load("./data/tmp/airtrafficmonth")
        .orderBy(col("month"));

    tc = System.currentTimeMillis();
    log.info("Reading gold zone in {} ms.", (tc - t0));
    t0 = tc;

    Dataset<Row> df = goldDf
        .withColumn("year", year(col("month")))
        .groupBy(col("year"))
        .agg(sum("pax").as("pax"),
            sum("internationalPax").as("internationalPax"),
            sum("domesticPax").as("domesticPax"))
        .orderBy(col("year"));

    tc = System.currentTimeMillis();
    log.info("Transformation for yearly graph {} ms.", (tc - t0));
    t0 = tc;

    // Precious processing time saved by using data in DL

    String[] inputCols = { "year" };
    VectorAssembler assembler = new VectorAssembler()
        .setInputCols(inputCols)
        .setOutputCol("features");
    df = assembler.transform(df);

    LinearRegression lr = new LinearRegression()
        .setMaxIter(10)
        .setRegParam(0.5)
        .setElasticNetParam(0.8)
        .setLabelCol("pax");

    int threshold = 2019;
    Dataset<Row> trainingData = df.filter(col("year").$less$eq(threshold));
    Dataset<Row> testData = df.filter(col("year").$greater(threshold));

    LinearRegressionModel model = lr.fit(trainingData);

    // Make predictions on test data, in this situation I have none
    Dataset<Row> predictions = model.transform(testData);
    predictions.show(20);

    Integer[] l =
        new Integer[] { 2020, 2021, 2022, 2023, 2024, 2025, 2026 };
    List<Integer> data = Arrays.asList(l);
    Dataset<Row> futuresDf =
        spark.createDataset(data, Encoders.INT()).toDF()
            .withColumnRenamed("value", "year");
    assembler = new VectorAssembler()
        .setInputCols(inputCols)
        .setOutputCol("features");
    futuresDf = assembler.transform(futuresDf);
    log.info("Futures");
    futuresDf.show();
    futuresDf.printSchema();
    log.info("/Futures");

    df = df.unionByName(futuresDf, true);

    // Graph
    df = model.transform(df);
    df.printSchema();
    df = df.drop("features");
    df = df.drop("rawFeatures");
    df = df.drop("internationalPax");
    df = df.drop("domesticPax");

    df = DrstiUtils.setHeader(df, "year", "Year");
    df = DrstiUtils.setHeader(df, "pax", "Passengers");
    df = DrstiUtils.setHeader(df, "prediction", "Prediction");

    DrstiLineChart d = new DrstiLineChart(df);
    d.setTitle("US air traffic, in passengers, per year");
    d.setXTitle("Year " + DataframeUtils.min(df, "year") + " - " +
        DataframeUtils.max(df, "year"));
    d.setYTitle("Passengers (000s)");
    d.render();

    tc = System.currentTimeMillis();
    log.info("Data exported for in {} ms.", (tc - t0));

    spark.stop();
    return true;
  }

  private double predict(double feature, RegressionModel model) {
    double p = model.predict(Vectors.dense(feature));
    log.info("Passengers in year {}: {}", feature, p);
    return p;
  }
}

package ai.jgp.drsti.spark.demo.airtraffic.lab600_monthly_air_traffic_delta;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.sum;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.jgp.drsti.spark.utils.DataframeUtils;

/**
 * 
 * @author jgp
 *
 */
public class MonthlyAirTrafficSaveDeltaWithImputationApp {
  private static Logger log = LoggerFactory
      .getLogger(MonthlyAirTrafficSaveDeltaWithImputationApp.class);

  public static void main(String[] args) {
    MonthlyAirTrafficSaveDeltaWithImputationApp app =
        new MonthlyAirTrafficSaveDeltaWithImputationApp();
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
        .appName("Pax with 2021 imputation to Delta")
        .master("local[*]")
        .getOrCreate();

    long tc = System.currentTimeMillis();
    log.info("Spark master available in {} ms.", (tc - t0));
    t0 = tc;

    // Creates the schema
    StructType schema = DataTypes.createStructType(new StructField[] {
        DataTypes.createStructField(
            "month",
            DataTypes.DateType,
            false),
        DataTypes.createStructField(
            "pax",
            DataTypes.IntegerType,
            true) });

    // International Pax
    Dataset<Row> internationalPaxDf = spark.read().format("csv")
        .option("header", true)
        .option("dateFormat", "MMMM yyyy")
        .schema(schema)
        .load(
            "data/bts/International USCarrier_Traffic_20210902163435.csv");
    internationalPaxDf = internationalPaxDf
        .withColumnRenamed("pax", "internationalPax")
        // Very simple data quality
        .filter(col("month").isNotNull())
        .filter(col("internationalPax").isNotNull());

    tc = System.currentTimeMillis();
    log.info("International pax ingested in {} ms.", (tc - t0));
    t0 = tc;

    // Domestic Pax
    Dataset<Row> domesticPaxDf = spark.read().format("csv")
        .option("header", true)
        .option("dateFormat", "MMMM yyyy")
        .schema(schema)
        .load(
            "data/bts/Domestic USCarrier_Traffic_20210902163435.csv");
    domesticPaxDf = domesticPaxDf
        .withColumnRenamed("pax", "domesticPax")
        // Very simple data quality
        .filter(col("month").isNotNull())
        .filter(col("domesticPax").isNotNull());
    tc = System.currentTimeMillis();
    log.info("Domestic pax ingested in {} ms.", (tc - t0));
    t0 = tc;

    // Combining datasets
    Dataset<Row> df = internationalPaxDf
        .join(domesticPaxDf,
            internationalPaxDf.col("month")
                .equalTo(domesticPaxDf.col("month")),
            "outer")
        .withColumn("pax", expr("internationalPax + domesticPax"))

        // Make a copy of the original BTS data
        .withColumn("paxBts", col("pax"))
        .drop(domesticPaxDf.col("month"))

        // Very simple data quality
        .orderBy(col("month"))
        .cache();
    tc = System.currentTimeMillis();
    log.info("Join in {} ms.", (tc - t0));
    t0 = tc;

    // Imputation of missing data
    Dataset<Row> df2021 =
        df.filter(expr(
            "month >= TO_DATE('2021-01-01') and month <= TO_DATE('2021-12-31')"));
    df2021.show();
    int monthCount = (int) df2021.count();
    log.info(
        "We only have {} months for 2021, let's impute the {} others.",
        monthCount, (12 - monthCount));

    df2021 = df2021
        .agg(sum("pax").as("pax"),
            sum("internationalPax").as("internationalPax"),
            sum("domesticPax").as("domesticPax"));
    int pax = DataframeUtils.maxAsInt(df2021, "pax") / (12 - monthCount);
    int intPax =
        DataframeUtils.maxAsInt(df2021, "internationalPax")
            / (12 - monthCount);
    int domPax =
        DataframeUtils.maxAsInt(df2021, "domesticPax") / (12 - monthCount);

    List<String> data = new ArrayList();
    for (int i = monthCount + 1; i <= 12; i++) {
      data.add("2021-" + i + "-01");
    }
    Dataset<Row> dfImputation2021 = spark
        .createDataset(data, Encoders.STRING()).toDF()
        .withColumn("month", col("value").cast(DataTypes.DateType))
        .withColumn("pax", lit(pax))
        .withColumn("internationalPax", lit(intPax))
        .withColumn("domesticPax", lit(domPax))
        .drop("value");
    log.info("Imputation done:");
    dfImputation2021.show();

    tc = System.currentTimeMillis();
    log.info("Imputation in {} ms.", (tc - t0));
    t0 = tc;

    // Combine data with imputated data
    df = df.unionByName(dfImputation2021, true);

    df.orderBy(col("month").desc()).show(15);
    df.write()
        .format("delta")
        .mode("overwrite")
        .save("./data/tmp/airtrafficmonth_all");

    tc = System.currentTimeMillis();
    log.info("Data save for in {} ms.", (tc - t0));

    spark.stop();
    return true;
  }
}

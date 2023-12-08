---
title: Advanced SQL Server Administration and Performance Tuning and Security
date: 2023-09-06 00:00:00 +0800
categories: [SQL, SQL_Server_Administration]
tags: [SQL]
---

# Advanced SQL Server Administration

## SQL Server Agent

SQL Server Agent is a Microsoft Windows service that executes scheduled administrative tasks, known as jobs.

### Creating a Job

```sql
USE msdb;
GO

EXEC msdb.dbo.sp_add_job
    @job_name = N'Job_Name',
    @enabled = 1;
GO

EXEC msdb.dbo.sp_add_jobstep
    @job_name = N'Job_Name',
    @step_name = N'Step_Name',
    @subsystem = N'T-SQL',
    @command = N'Your_T-SQL_Command';
GO
```

### Scheduling a Job

```sql
USE msdb;
GO

EXEC msdb.dbo.sp_add_schedule
    @schedule_name = N'Schedule_Name',
    @freq_type = 1,
    @active_start_time = 100000;
GO

EXEC msdb.dbo.sp_attach_schedule
    @job_name = N'Job_Name',
    @schedule_name = N'Schedule_Name';
GO
```

## SQL Server Integration Services (SSIS)

SSIS is a part of SQL Server and is used for solving complex business problems by copying or downloading files, extracting and transforming data from different sources, and loading data into one or several destinations.

### Creating an SSIS Package

You can create SSIS packages using SQL Server Data Tools (SSDT).

## Replication

Replication is a set of technologies for copying and distributing data and database objects from one database to another.

### Transactional Replication

Transactional replication is typically used in server-to-server environments and is suitable for high-volume environments.

### Merge Replication

Merge replication allows updates to be made at both the publisher and the subscriber.

## Always On Availability Groups

Always On Availability Groups is a high-availability and disaster-recovery solution that provides an enterprise-level alternative to database mirroring.

### Creating an Availability Group

```sql
CREATE AVAILABILITY GROUP AG_Name
WITH
    (   
        REPLICA ON 'Primary_Server' WITH
        (
            ENDPOINT_URL = 'TCP://Primary_Server:5022',
            AVAILABILITY_MODE = ASYNCHRONOUS_COMMIT,
            FAILOVER_MODE = MANUAL,
            SEEDING_MODE = MANUAL
        )
    );
GO
```

# Advanced SQL Server Performance Tuning

## Query Performance Tuning

### Execution Plan Analysis

Use the SQL Server Management Studio (SSMS) to analyze execution plans and identify performance bottlenecks.

### Index Maintenance

Regularly monitor and maintain indexes for optimal query performance.

#### Rebuilding Indexes

```sql
ALTER INDEX index_name ON table_name REBUILD;
```

#### Updating Statistics

```sql
UPDATE STATISTICS table_name;
```

## In-Memory OLTP

In-Memory OLTP is a feature that enables you to create memory-optimized tables for high-performance transaction processing.

### Creating an In-Memory Table

```sql
CREATE TABLE dbo.MemoryOptimizedTable
(
    ID INT NOT NULL PRIMARY KEY NONCLUSTERED,
    Name NVARCHAR(50) NOT NULL,
    INDEX IX_Name NONCLUSTERED HASH (ID) WITH (BUCKET_COUNT = 1024)
) WITH (MEMORY_OPTIMIZED = ON);
```

### Natively Compiled Stored Procedures

```sql
CREATE PROCEDURE dbo.NativeCompiledProc
WITH NATIVE_COMPILATION, SCHEMABINDING
AS BEGIN ATOMIC WITH (TRANSACTION ISOLATION LEVEL = SNAPSHOT, LANGUAGE = N'us_english')
    -- SQL statements
END;
```

## Temporal Tables

Temporal tables allow you to keep a history of data changes in a table.

### Creating a Temporal Table

```sql
CREATE TABLE dbo.Employee
(
    EmployeeID INT PRIMARY KEY CLUSTERED,
    Name NVARCHAR(100),
    Salary INT,
    SysStartTime datetime2 GENERATED ALWAYS AS ROW START HIDDEN NOT NULL,
    SysEndTime datetime2 GENERATED ALWAYS AS ROW END HIDDEN NOT NULL,
    PERIOD FOR SYSTEM_TIME (SysStartTime, SysEndTime)
)
WITH (SYSTEM_VERSIONING = ON);
```

## SQL Server on Linux

SQL Server is now available on Linux, providing flexibility in deployment.

### Installing SQL Server on Linux


# Advanced SQL Server Security

## Always Encrypted

Always Encrypted is a feature designed to protect sensitive data, such as credit card numbers or social security numbers.

### Creating a Table with Always Encrypted

```sql
CREATE TABLE dbo.SensitiveData
(
    ID INT PRIMARY KEY,
    CreditCardNumber NVARCHAR(25) ENCRYPTED WITH (COLUMN_ENCRYPTION_KEY = EncKey, ENCRYPTION_TYPE = Deterministic, ALGORITHM = 'AEAD_AES_256_CBC_HMAC_SHA_256')
);
```

### Managing Always Encrypted Keys

Use SQL Server Management Studio (SSMS) or PowerShell to manage keys.

## Dynamic Data Masking

Dynamic Data Masking limits sensitive data exposure by masking it to non-privileged users.

### Applying Masking to a Column

```sql
ALTER TABLE dbo.SensitiveData
ALTER COLUMN CreditCardNumber ADD MASKED WITH (FUNCTION = 'partial(4, "XXXX", 4)');
```

## Transparent Data Encryption (TDE)

TDE encrypts the database files at rest to protect against unauthorized access.

### Enabling TDE

```sql
USE master;
GO

CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'YourPassword';
CREATE CERTIFICATE TDECert WITH SUBJECT = 'TDE Certificate';
CREATE DATABASE ENCRYPTION KEY
WITH ALGORITHM = AES_256
ENCRYPTION BY SERVER CERTIFICATE TDECert;
ALTER DATABASE YourDatabase SET ENCRYPTION ON;
```

## SQL Server Auditing

SQL Server Auditing tracks database events and writes the events to the audit log.

### Creating an Audit

```sql
CREATE SERVER AUDIT ServerAudit
TO FILE (FILEPATH = 'C:\SQLServer\Audit\')
WITH (ON_FAILURE = CONTINUE);
```

### Enabling the Audit

```sql
ALTER SERVER AUDIT ServerAudit WITH (STATE = ON);
```

These advanced topics in security, including Always Encrypted, Dynamic Data Masking, Transparent Data Encryption, and SQL Server Auditing, will help you implement robust security measures in your SQL Server environment. 





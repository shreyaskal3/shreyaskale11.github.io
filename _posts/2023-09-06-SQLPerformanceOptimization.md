---
title: SQL Performance Optimization
date: 2023-09-06 00:00:00 +0800
categories: [SQL, Basic_SQL]
tags: [SQL]
---

# Performance Optimization

## Indexing Strategies

### Clustered Index

Defines the order in which data is physically stored in a table.

```sql
CREATE CLUSTERED INDEX idx_clustered ON table (column);
```

### Non-Clustered Index

Creates a separate structure for indexing, allowing multiple indexes per table.

```sql
CREATE NONCLUSTERED INDEX idx_nonclustered ON table (column);
```

### Covering Index

Includes all the columns required to satisfy a query, eliminating the need to access the actual table.

```sql
CREATE INDEX idx_covering ON table (column1, column2) INCLUDE (column3);
```

## Query Optimization

### Execution Plan

Review the execution plan to analyze how SQL Server processes a query.

```sql
EXPLAIN SELECT column1, column2 FROM table WHERE condition;
```

### Query Hints

Optimize queries by providing hints to the query optimizer.

```sql
SELECT column1, column2
FROM table WITH (INDEX(idx_nonclustered), FORCESEEK)
WHERE condition;
```

## Backup and Restore

### Full Backup

```sql
BACKUP DATABASE database_name TO DISK = 'path\to\backup\file.bak';
```

### Restore Database

```sql
RESTORE DATABASE database_name FROM DISK = 'path\to\backup\file.bak';
```

## Security

### Creating a User

```sql
CREATE LOGIN login_name WITH PASSWORD = 'password';
CREATE USER user_name FOR LOGIN login_name;
```

### Granting Permissions

```sql
GRANT SELECT, INSERT ON table TO user_name;
```
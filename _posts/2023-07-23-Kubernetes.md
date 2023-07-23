---
title: CoreDNS and Kube-DNS in Kubernetes
date: 2023-07-23 00:00:00 +0800
categories: [Blogging, Tutorial]
tags: [favicon]
---

What are main differences between CoreDNS and Kube-DNS in Kubernetes ?

🪄Caching
- CoreDNS uses its own built-in caching, while KubeDNS uses dnsmasq. CoreDNS's caching is multi-threaded, while dnsmasq is single-threaded. This means that CoreDNS can handle more requests per second than KubeDNS.

🔄 Flexibility and Plugin Architecture
- CoreDNS: Highly flexible with plugin support ⚙️
- Kube-DNS: Monolithic architecture, limited customization 🚫

🚀 Deployment and Integration
- CoreDNS: Default DNS server from Kubernetes 1.27 onwards 🆕
- Kube-DNS: Original DNS solution, phased out in favor of CoreDNS 🚫

⚡ Performance and Efficiency
- CoreDNS: Better performance, lower resource utilization 💪
- Kube-DNS: Some limitations in scalability and performance 🐌

🔒 Protocol Support
- CoreDNS: Supports DNS-over-TLS and DNS-over-HTTPS 🔒
- Kube-DNS: No support for DNS-over-TLS or DNS-over-HTTPS 🚫

🚧 Deprecation
- CoreDNS: Gradually replaced Kube-DNS, actively developed and recommended 🔄
- Kube-DNS: Deprecated and no longer actively developed 🚫

Image credit: Labyirnth
#kubernetes #cka #ckad #gke #google #googlecloud #aws #cncf #k8s
Activate to view larger image,
diagram labyrinth

![image](https://media.licdn.com/dms/image/D4D22AQFqkMdNstRV0g/feedshare-shrink_800/0/1689833123739?e=1692835200&v=beta&t=dJ9LPUKprW5OGEMMiEMYExo4Yill49uZp0DpM5ASVQ8)








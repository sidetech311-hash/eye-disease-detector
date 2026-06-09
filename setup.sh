#!/bin/bash
mkdir -p ~/.streamlit/lib
wget -q -O ~/.streamlit/lib/libgthread-2.0.so.0 \
  http://ftp.debian.org/debian/pool/main/g/glib2.0/libglib2.0-0_2.74.6-2_amd64.deb
dpkg-deb -x ~/.streamlit/lib/libglib2.0-0_2.74.6-2_amd64.deb /tmp/glib_extract
cp /tmp/glib_extract/usr/lib/x86_64-linux-gnu/libgthread-2.0.so.0 ~/.streamlit/lib/
echo "libgthread installed"

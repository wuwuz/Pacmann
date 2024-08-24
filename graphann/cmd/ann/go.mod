module main

go 1.22.1

replace example.com/graphann => ../../../graphann

require example.com/graphann v0.0.0-00010101000000-000000000000

require (
	github.com/evan176/hnswgo v0.0.0-20220622031020-39253a76f9e4 // indirect
	github.com/kshard/fvecs v0.0.1 // indirect
	github.com/kshedden/gonpy v0.0.0-20210519231815-fa3c8dd8e59b // indirect
	github.com/yahoojapan/gongt v0.0.0-20190517050727-966dcc7aa5e8 // indirect
)

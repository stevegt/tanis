package node

import (
	"sync"

	. "github.com/stevegt/goadapt"

	_ "net/http/pprof"

	"github.com/emicklei/dot"
)

// Log collects messages in a channel and writes them to stdout.
type Log struct {
	MsgChan chan string
}

// NewLog creates a new Log.
func NewLog() (l *Log) {
	l = &Log{
		MsgChan: make(chan string, 99999),
	}
	go func() {
		for msg := range l.MsgChan {
			Pl(msg)
		}
	}()
	return
}

// I logs a message.
func I(args ...interface{}) {
	msg := FormatArgs(args...)
	logger.MsgChan <- msg
}

var logger *Log


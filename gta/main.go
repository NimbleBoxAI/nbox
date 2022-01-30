package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os/exec"
	"time"
)

type Test struct {
	name       string
	start      time.Time
	end        time.Time
	duration   time.Time
	packge_dep []string
}

func main() {

	// We'll start with a simple command that takes no
	// arguments or input and just prints something to
	// stdout. The `exec.Command` helper creates an object
	// to represent this external process.
	// `.Output` is another helper that handles the common
	// case of running a command, waiting for it to finish,
	// and collecting its output. If there were no errors,
	// `dateOut` will hold bytes with the date info.
	// dateCmd := exec.Command("date")
	// dateOut, err := dateCmd.Output()
	// if err != nil {
	// 	panic(err)
	// }
	log.Println("Starting Tests ...")
	log.Println("[1/2] Getting dependencies")

	freeze_command := exec.Command("python3", "-m", "pip", "freeze")
	freeze_out, _ := freeze_command.StdoutPipe()
	freeze_command.Start()
	grepBytes, _ := ioutil.ReadAll(freeze_out)
	fmt.Print(string(grepBytes))

	// // Next we'll look at a slightly more involved case
	// // where we pipe data to the external process on its
	// // `stdin` and collect the results from its `stdout`.
	// grepCmd := exec.Command("grep", "hello")

	// // Here we explicitly grab input/output pipes, start
	// // the process, write some input to it, read the
	// // resulting output, and finally wait for the process
	// // to exit.
	// grepIn, _ := grepCmd.StdinPipe()
	// grepOut, _ := grepCmd.StdoutPipe()
	// grepCmd.Start()
	// fmt.Println("Writing in terminal\n", ">>> hello grep\ngoodbye grep")

	// grepIn.Write([]byte("hello grep\ngoodbye grep"))
	// grepIn.Close()
	// grepBytes, _ := ioutil.ReadAll(grepOut)
	// grepCmd.Wait()

	// // We ommited error checks in the above example, but
	// // you could use the usual `if err != nil` pattern for
	// // all of them. We also only collect the `StdoutPipe`
	// // results, but you could collect the `StderrPipe` in
	// // exactly the same way.
	// fmt.Println("> grep hello")
	// fmt.Println(string(grepBytes))
	// fmt.Println("-------------------")

	// // Note that when spawning commands we need to
	// // provide an explicitly delineated command and
	// // argument array, vs. being able to just pass in one
	// // command-line string. If you want to spawn a full
	// // command with a string, you can use `bash`'s `-c`
	// // option:
	// lsCmd := exec.Command("bash", "-c", "ls -a -l -h")
	// lsOut, err := lsCmd.Output()
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println("> ls -a -l -h")
	// fmt.Println(string(lsOut))
}
